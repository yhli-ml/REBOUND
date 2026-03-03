#!/usr/bin/env python3
"""LoRA fine-tuning of Stable Diffusion on a long-tail dataset.

Fine-tunes the UNet of Stable Diffusion using LoRA adapters so that
the diffusion model learns the visual distribution of the target dataset
(e.g., CIFAR-100-LT style, resolution, color distribution).

This step is OPTIONAL but recommended for better domain consistency.

Key design choices:
  - Only trains LoRA layers (rank=4) → ~2MB of parameters
  - Uses all classes (head + tail) for training
  - Simple captions: "a photo of a {class_name}"
  - Trains for a few epochs only to avoid overfitting

Usage:
  python -m augment.head2tail_lora_finetune \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./lora_weights/cifar100_lt \\
      --train_steps 2000 --lr 1e-4

After fine-tuning, pass the LoRA weights to the generation pipeline:
  python generate_head2tail.py --lora_weights ./lora_weights/cifar100_lt
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='LoRA fine-tuning of SD on long-tail dataset')

    parser.add_argument('--dataset', type=str, default='cifar100_lt',
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--imb_factor', type=float, default=0.01)

    parser.add_argument('--model_id', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='Base SD model to fine-tune')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save LoRA weights')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Training resolution')

    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=4.0,
                        help='LoRA alpha (scaling)')

    # Training
    parser.add_argument('--train_steps', type=int, default=2000,
                        help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


class TextImageDataset(Dataset):
    """Simple dataset that pairs images with "a photo of a {class}" captions.

    Handles both CIFAR (numpy arrays) and ImageNet (file paths).
    """

    def __init__(self, data, targets, class_names, resolution=512):
        self.data = data
        self.targets = targets
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.targets[idx]

        if isinstance(item, np.ndarray):
            image = Image.fromarray(item).convert('RGB')
        elif isinstance(item, str):
            image = Image.open(item).convert('RGB')
        else:
            image = item.convert('RGB')

        image = self.transform(image)
        class_name = self.class_names[label].replace('_', ' ')
        caption = f"a photo of a {class_name}"

        return {"image": image, "caption": caption}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("=" * 60)
    print("LoRA Fine-tuning of Stable Diffusion")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print()

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from augment.head2tail_prompts import get_class_names

    class_names = get_class_names(args.dataset)
    is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')

    if is_cifar:
        from datasets.cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100
        if args.dataset == 'cifar10_lt':
            ds = IMBALANCECIFAR10(root=args.data_root, train=True,
                                  imb_factor=args.imb_factor, download=True)
        else:
            ds = IMBALANCECIFAR100(root=args.data_root, train=True,
                                   imb_factor=args.imb_factor, download=True)
        data = ds.data
        targets = list(ds.targets)
    else:
        from datasets import get_dataset
        img_root = args.img_root if args.img_root else None
        ds = get_dataset(args.dataset, args.data_root, train=True,
                         transform=None, img_root=img_root)
        data = ds.img_paths
        targets = list(ds.targets)

    print(f"Loaded {len(data)} training samples")

    train_ds = TextImageDataset(data, targets, class_names,
                                resolution=args.resolution)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    # Load models
    print("\nLoading Stable Diffusion components...")
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.float32
    ).to(args.device)
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=torch.float32
    ).to(args.device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # Freeze everything except LoRA
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    print("Adding LoRA adapters to UNet...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.lr, weight_decay=1e-2
    )

    # Training loop
    print(f"\nStarting LoRA fine-tuning for {args.train_steps} steps...")
    unet.train()
    text_encoder.eval()
    vae.eval()

    global_step = 0
    data_iter = iter(train_loader)

    while global_step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(args.device)
        captions = batch["caption"]

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Encode text
        text_input = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids)[0]

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=args.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        loss = loss / args.grad_accum
        loss.backward()

        if (global_step + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % 100 == 0:
            print(f"  Step {global_step}/{args.train_steps}, Loss: {loss.item():.4f}")

        if global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
            unet.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint to {ckpt_dir}")

    # Save final LoRA weights
    final_dir = os.path.join(args.output_dir, 'final')
    unet.save_pretrained(final_dir)
    print(f"\nFinal LoRA weights saved to {final_dir}")
    print("LoRA fine-tuning complete!")


if __name__ == '__main__':
    main()
