#!/usr/bin/env python3
"""Offline DiffuseMix reference baseline for long-tail datasets.

Generates augmented images using the DiffuseMix pipeline
(InstructPix2Pix style transfer + concatenation + fractal blending).

This implementation follows the original DiffuseMix generation plan:
each original training image produces `m` augmented images, with one prompt
sampled from the prompt pool for each generated image.

Usage examples:
  # CIFAR-100-LT: generate one DiffuseMix sample per original image
  python generate_diffusemix.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --fractal_dir /path/to/deviantart \\
      --prompts "sunset,Autumn,watercolor art" \\
      --aug_per_image 1 --output_dir ./data/diffusemix_cifar100_lt \\
      --gen_size 256

  # ImageNet-LT: generate two DiffuseMix samples per original image
  python generate_diffusemix.py \\
      --dataset imagenet_lt \\
      --data_root ./data/ImageNet_LT \\
      --img_root /path/to/imagenet \\
      --fractal_dir /path/to/deviantart \\
      --prompts "sunset,Autumn" \\
      --aug_per_image 2 --output_dir ./data/diffusemix_imagenet_lt

Output structure:
  output_dir/
    metadata.json          # {class_label: [{"path": "...", "label": int}, ...]}
    generation_config.json
    class_000/
      aug_00000_sunset.jpg
      aug_00001_Autumn.jpg
      ...
    class_001/
      ...
"""

import argparse
import hashlib
import json
import os
import sys
import time
import random
import numpy as np
from PIL import Image

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='DiffuseMix reference baseline for long-tail datasets')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt',
                                 'inaturalist', 'places_lt'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--img_root', type=str, default='',
                        help='Image root (for ImageNet-LT, iNaturalist, Places-LT)')
    parser.add_argument('--imb_factor', type=float, default=0.01,
                        help='Imbalance factor for CIFAR (1/IF)')

    # DiffuseMix
    parser.add_argument('--fractal_dir', type=str, required=True,
                        help='Directory containing fractal images (deviantart)')
    parser.add_argument('--prompts', type=str, default='sunset,Autumn',
                        help='Comma-separated style prompts')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                        help='InstructPix2Pix guidance scale')
    parser.add_argument('--fractal_alpha', type=float, default=0.20,
                        help='Fractal blending alpha')
    parser.add_argument('--gen_size', type=int, default=256,
                        help='Image size for generation')
    parser.add_argument('--model_id', type=str,
                        default='timbrooks/instruct-pix2pix',
                        help='HuggingFace model ID for InstructPix2Pix')

    # Generation plan
    parser.add_argument('--plan_mode', type=str, default='per_image',
                        choices=['per_image'],
                        help='Generation plan. DiffuseMix uses per-image augmentation.')
    parser.add_argument('--aug_per_image', type=int, default=1,
                        help='Number of DiffuseMix samples generated per original image')

    # Deprecated legacy planning arguments. Kept for CLI compatibility only.
    parser.add_argument('--target_num', type=int, default=-1,
                        help='Deprecated legacy argument. Ignored because '
                             'DiffuseMix now uses per-image generation.')
    parser.add_argument('--max_aug_per_class', type=int, default=500,
                        help='Deprecated legacy argument. Ignored.')
    parser.add_argument('--min_class_count', type=int, default=0,
                        help='Deprecated legacy argument. Ignored.')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save augmented images')
    parser.add_argument('--save_size', type=int, default=0,
                        help='Save size (0 = same as gen_size). '
                             'For CIFAR, set to 32 to save at original resolution')

    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Not used (generation is sequential on GPU)')

    return parser.parse_args()


def get_dataset_samples(args):
    """Load dataset and return list of (image_or_path, label, class_name) tuples.

    For CIFAR: images are numpy arrays (H, W, C).
    For ImageNet-LT etc.: images are file paths.

    Returns:
        samples: list of (image_or_path, label_int)
        cls_num_list: list of per-class counts
        num_classes: int
        is_cifar: bool
    """
    is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')

    if is_cifar:
        sys.path.insert(0, os.path.dirname(__file__))
        from datasets.cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100

        if args.dataset == 'cifar10_lt':
            ds = IMBALANCECIFAR10(root=args.data_root, train=True,
                                  imb_factor=args.imb_factor, download=True)
            num_classes = 10
        else:
            ds = IMBALANCECIFAR100(root=args.data_root, train=True,
                                   imb_factor=args.imb_factor, download=True)
            num_classes = 100

        cls_num_list = ds.cls_num_list
        # samples: (numpy_array_HWC, label)
        samples = [(ds.data[i], ds.targets[i]) for i in range(len(ds.data))]
        return samples, cls_num_list, num_classes, True

    else:
        sys.path.insert(0, os.path.dirname(__file__))
        from datasets import get_dataset

        img_root = args.img_root if args.img_root else None
        ds = get_dataset(args.dataset, args.data_root, train=True,
                        transform=None, img_root=img_root)
        cls_num_list = ds.cls_num_list

        if args.dataset == 'imagenet_lt':
            num_classes = 1000
            # samples: (file_path, label)
            samples = list(zip(ds.img_paths, ds.targets))
        elif args.dataset == 'inaturalist':
            num_classes = 8142
            samples = list(zip(ds.img_paths, ds.targets))
        elif args.dataset == 'places_lt':
            num_classes = 365
            samples = list(zip(ds.img_paths, ds.targets))
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        return samples, cls_num_list, num_classes, False


def compute_per_image_plan(cls_num_list, aug_per_image):
    """Compute per-image DiffuseMix generation counts for every class."""
    return {
        class_idx: count * aug_per_image
        for class_idx, count in enumerate(cls_num_list)
        if count > 0 and aug_per_image > 0
    }


def warn_if_legacy_plan_args_used(args):
    legacy_args = []
    if args.target_num != -1:
        legacy_args.append('--target_num')
    if args.max_aug_per_class != 500:
        legacy_args.append('--max_aug_per_class')
    if args.min_class_count != 0:
        legacy_args.append('--min_class_count')
    if legacy_args:
        print("[WARN] Ignoring deprecated legacy planning arguments: "
              + ", ".join(legacy_args))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
    if not prompts:
        raise ValueError("At least one non-empty prompt is required.")
    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size) if args.save_size > 0 else gen_size

    print("=" * 60)
    print("DiffuseMix Reference Baseline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Prompts: {prompts}")
    print(f"Plan mode: {args.plan_mode} (m={args.aug_per_image} per image)")
    print(f"Output: {args.output_dir}")
    print()
    warn_if_legacy_plan_args_used(args)

    # Load dataset
    print("[1/5] Loading dataset...")
    samples, cls_num_list, num_classes, is_cifar = get_dataset_samples(args)
    print(f"  Classes: {num_classes}, Total samples: {len(samples)}")
    print(f"  Max class count: {max(cls_num_list)}, Min: {min(cls_num_list)}")

    # Organize samples by class
    class_samples = {c: [] for c in range(num_classes)}
    for item, label in samples:
        class_samples[label].append(item)

    # Compute augmentation plan
    aug_plan = compute_per_image_plan(cls_num_list, args.aug_per_image)

    total_to_generate = sum(aug_plan.values())
    classes_to_augment = len(aug_plan)
    print(f"\n[2/5] Augmentation plan:")
    print(f"  Classes to augment: {classes_to_augment}/{num_classes}")
    print(f"  Total images to generate: {total_to_generate}")
    if aug_plan:
        counts = list(aug_plan.values())
        print(f"  Per-class range: {min(counts)} ~ {max(counts)}")
    print()

    if total_to_generate == 0:
        print("Nothing to augment. Exiting.")
        return

    # Load fractal images
    print("[3/5] Loading fractal images...")
    from augment.diffusemix_utils import load_fractal_images
    fractal_imgs = load_fractal_images(args.fractal_dir, size=gen_size)

    # Load diffusion model
    print("[4/5] Loading InstructPix2Pix model...")
    from augment.diffusemix_handler import ModelHandler
    model = ModelHandler(model_id=args.model_id, device=args.device)

    from augment.diffusemix_utils import (
        combine_images, blend_with_fractal, is_black_image
    )

    # Generate augmented images
    print(f"\n[5/5] Generating augmented images...")
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {}  # {class_str: [{"path": ..., "label": int}, ...]}
    global_count = 0
    start_time = time.time()

    for class_idx in sorted(aug_plan.keys()):
        n_to_gen = aug_plan[class_idx]
        class_dir = os.path.join(args.output_dir, f'class_{class_idx:04d}')
        os.makedirs(class_dir, exist_ok=True)

        class_key = f'class_{class_idx:04d}'
        metadata[class_key] = []

        class_original = class_samples[class_idx]
        n_original = len(class_original)

        print(f"  Class {class_idx:4d}: {n_original} original, "
              f"generating {n_to_gen} augmented...")

        for original_idx, item in enumerate(class_original):
            # Prepare original as PIL once, then reuse it for all augmentations.
            if is_cifar:
                original_pil = Image.fromarray(item).convert('RGB')
            else:
                original_pil = Image.open(item).convert('RGB')
            original_resized = original_pil.resize(gen_size)

            for aug_idx in range(args.aug_per_image):
                generated = False

                for attempt_idx in range(3):
                    prompt = rng.choice(prompts)

                    try:
                        gen_imgs = model.generate_images_from_pil(
                            prompt, original_resized, num_images=1,
                            guidance_scale=args.guidance_scale, size=gen_size)
                    except Exception as e:
                        print(f"    [WARN] Generation failed for class {class_idx} "
                              f"sample {original_idx} aug {aug_idx}: {e}")
                        continue

                    gen_img = gen_imgs[0].resize(gen_size)
                    if is_black_image(gen_img):
                        continue

                    # Step 2: Concatenate
                    combined = combine_images(original_resized, gen_img)

                    # Step 3: Fractal blend
                    fractal = rng.choice(fractal_imgs)
                    blended = blend_with_fractal(combined, fractal, args.fractal_alpha)

                    # Save
                    if save_size != gen_size:
                        blended = blended.resize(save_size)

                    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
                    fname = (
                        f'aug_{original_idx:05d}_aug{aug_idx}_{prompt_hash}.jpg'
                    )
                    fpath = os.path.join(class_dir, fname)
                    blended.save(fpath, quality=95)

                    metadata[class_key].append({
                        'path': os.path.join(class_key, fname),
                        'label': class_idx,
                        'aug_type': 'diffusemix',
                        'plan_mode': args.plan_mode,
                        'prompt': prompt,
                        'source_sample_index': original_idx,
                        'attempt_index': attempt_idx,
                    })
                    global_count += 1
                    generated = True

                    if global_count % 100 == 0:
                        elapsed = time.time() - start_time
                        speed = global_count / elapsed
                        remaining = (total_to_generate - global_count) / max(speed, 1e-6)
                        print(f"    Progress: {global_count}/{total_to_generate} "
                              f"({speed:.1f} img/s, ETA: {remaining/60:.1f} min)")
                    break

                if not generated:
                    print(f"    [WARN] Skipped sample {original_idx} aug {aug_idx} "
                          f"for class {class_idx} after repeated failures.")

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save a simple txt file: path label
    txt_path = os.path.join(args.output_dir, 'augmented_list.txt')
    with open(txt_path, 'w') as f:
        for class_key in sorted(metadata.keys()):
            for entry in metadata[class_key]:
                f.write(f"{entry['path']} {entry['label']}\n")

    config = {
        'method': 'diffusemix',
        'dataset': args.dataset,
        'imb_factor': args.imb_factor,
        'plan_mode': args.plan_mode,
        'aug_per_image': args.aug_per_image,
        'prompts': prompts,
        'guidance_scale': args.guidance_scale,
        'fractal_alpha': args.fractal_alpha,
        'model_id': args.model_id,
        'gen_size': args.gen_size,
        'save_size': args.save_size,
        'total_generated': global_count,
        'seed': args.seed,
    }
    config_path = os.path.join(args.output_dir, 'generation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {global_count} augmented images in {elapsed/60:.1f} min")
    print(f"Metadata: {meta_path}")
    print(f"Image list: {txt_path}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
