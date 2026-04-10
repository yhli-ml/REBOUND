#!/usr/bin/env python3
"""Bare prompt-controlled diffusion augmentation baseline.

This is a deliberately simple reference method for checking whether Head2Tail's
head-to-tail transfer is doing more than plain class-name prompting.

Modes:
  - txt2img: generate from text only, prompt = "a photo of a {class_name}"
  - img2img: edit same-class real images with the same minimal class prompt

Outputs are compatible with DiffuseMixDataset / main.py --diffusemix_dir.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from augment.head2tail_prompts import get_class_names
from utils import get_class_split_lists


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bare prompt-controlled diffusion augmentation baseline')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--imb_factor', type=float, default=0.01)

    parser.add_argument('--generation_mode', type=str, default='txt2img',
                        choices=['txt2img', 'img2img'],
                        help='txt2img = prompt-only generation; '
                             'img2img = same-class real image init + prompt')
    parser.add_argument('--model_id', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--prompt_template', type=str,
                        default='a photo of a {class_name}',
                        help='Minimal class prompt template')
    parser.add_argument('--negative_prompt', type=str,
                        default='blurry, low quality, artifacts, distorted, '
                                'watermark, text, deformed')
    parser.add_argument('--strength', type=float, default=0.6,
                        help='Only used for img2img')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--gen_size', type=int, default=512)

    parser.add_argument('--per_image_scope', type=str, default='medium_tail',
                        choices=['all', 'medium_tail'],
                        help='Which classes to augment')
    parser.add_argument('--aug_per_image', type=int, default=1,
                        help='Number of generated images per selected real image')
    parser.add_argument('--head_threshold', type=int, default=100)
    parser.add_argument('--tail_threshold', type=int, default=20)

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_size', type=int, default=0,
                        help='0 = 32 for CIFAR, gen_size otherwise')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    if args.save_size == 0:
        args.save_size = 32 if args.dataset in ('cifar10_lt', 'cifar100_lt') else args.gen_size
    if args.aug_per_image <= 0:
        raise ValueError('--aug_per_image must be positive')
    return args


def load_dataset_samples(args):
    is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')

    if is_cifar:
        from datasets.cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100
        if args.dataset == 'cifar10_lt':
            ds = IMBALANCECIFAR10(root=args.data_root, train=True,
                                  imb_factor=args.imb_factor, download=True)
            num_classes = 10
        else:
            ds = IMBALANCECIFAR100(root=args.data_root, train=True,
                                   imb_factor=args.imb_factor, download=True)
            num_classes = 100
        class_samples = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(ds.targets):
            class_samples[int(label)].append(ds.data[idx])
        return class_samples, list(ds.cls_num_list), num_classes, True

    from datasets import get_dataset
    img_root = args.img_root if args.img_root else None
    ds = get_dataset(args.dataset, args.data_root, train=True,
                     transform=None, img_root=img_root)
    num_classes = 1000
    class_samples = {c: [] for c in range(num_classes)}
    for path, label in zip(ds.img_paths, ds.targets):
        class_samples[int(label)].append(path)
    return class_samples, list(ds.cls_num_list), num_classes, False


def compute_generation_plan(cls_num_list, aug_per_image, per_image_scope,
                            head_threshold, tail_threshold, dataset_name):
    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )
    head_set = set(head_classes)

    plan = {}
    for class_idx, n_real in enumerate(cls_num_list):
        if n_real <= 0:
            continue
        if per_image_scope == 'medium_tail' and class_idx in head_set:
            continue
        plan[class_idx] = n_real * aug_per_image

    return plan, head_classes, medium_classes, tail_classes


def load_pipeline(args):
    if args.generation_mode == 'txt2img':
        from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(args.device)
    else:
        from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(args.device)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def to_pil_image(sample):
    if isinstance(sample, np.ndarray):
        return Image.fromarray(sample).convert('RGB')
    if isinstance(sample, str):
        return Image.open(sample).convert('RGB')
    return sample.convert('RGB')


def is_valid_image(image, threshold=0.9):
    arr = np.array(image.convert('L'))
    total = arr.size
    if (arr < 10).sum() > threshold * total:
        return False
    if (arr > 245).sum() > threshold * total:
        return False
    return True


def safe_name(name):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)[:80]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size)

    print("=" * 60)
    print("Bare Prompt-Controlled Diffusion Baseline")
    print("=" * 60)
    print(f"Dataset: {args.dataset} (imb_factor={args.imb_factor})")
    print(f"Mode: {args.generation_mode}")
    print(f"Scope: {args.per_image_scope}, m={args.aug_per_image}")
    print(f"Model: {args.model_id} (no LoRA)")
    print(f"Output: {args.output_dir}")

    class_samples, cls_num_list, num_classes, is_cifar = load_dataset_samples(args)
    class_names = get_class_names(args.dataset)
    plan, head_classes, medium_classes, tail_classes = compute_generation_plan(
        cls_num_list,
        args.aug_per_image,
        args.per_image_scope,
        args.head_threshold,
        args.tail_threshold,
        args.dataset,
    )

    print(f"Classes: {num_classes}, train samples: {sum(cls_num_list)}")
    print(f"Split: head={len(head_classes)}, medium={len(medium_classes)}, tail={len(tail_classes)}")
    print(f"Planned generated images: {sum(plan.values())}")

    pipe = load_pipeline(args)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    metadata = defaultdict(list)
    augmented_list = []
    total_generated = 0
    start = time.time()

    for class_idx in sorted(plan):
        class_name = class_names[class_idx].replace('_', ' ')
        prompt = args.prompt_template.format(class_name=class_name)
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
        class_dir_key = f'class_{class_idx:04d}'
        class_dir = os.path.join(args.output_dir, class_dir_key)
        os.makedirs(class_dir, exist_ok=True)

        n_to_gen = plan[class_idx]
        samples = class_samples[class_idx]
        generated = 0
        attempts = 0
        max_attempts = max(n_to_gen * 3, n_to_gen + 5)

        print(f"[Class {class_idx:04d}] {class_name}: generating {n_to_gen}")
        while generated < n_to_gen and attempts < max_attempts:
            attempts += 1

            if args.generation_mode == 'txt2img':
                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    height=args.gen_size,
                    width=args.gen_size,
                    generator=generator,
                )
                source_sample_index = None
            else:
                source_sample_index = generated % len(samples)
                init_image = to_pil_image(samples[source_sample_index]).resize(gen_size)
                result = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=args.strength,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                )

            image = result.images[0]
            if not is_valid_image(image):
                print(f"  [WARN] invalid image at attempt {attempts}, retrying")
                continue

            if save_size != gen_size:
                image = image.resize(save_size, Image.BICUBIC)

            fname = (
                f"bare_{args.generation_mode}_{generated:05d}_"
                f"{safe_name(class_name)}_{prompt_hash}.jpg"
            )
            rel_path = os.path.join(class_dir_key, fname)
            image.save(os.path.join(args.output_dir, rel_path), quality=95)

            entry = {
                'path': rel_path,
                'label': class_idx,
                'class_name': class_name,
                'prompt': prompt,
                'generation_mode': args.generation_mode,
                'source_sample_index': source_sample_index,
                'aug_type': 'bare_prompt_diffusion',
            }
            metadata[class_dir_key].append(entry)
            augmented_list.append(f"{rel_path} {class_idx}\n")
            generated += 1
            total_generated += 1

        if generated < n_to_gen:
            print(f"  [WARN] only generated {generated}/{n_to_gen} for class {class_idx}")

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(dict(metadata), f, indent=2)
    with open(os.path.join(args.output_dir, 'augmented_list.txt'), 'w') as f:
        f.writelines(augmented_list)

    config = vars(args).copy()
    config.update({
        'method': 'bare_prompt_diffusion',
        'total_generated': total_generated,
        'num_classes': num_classes,
        'head_classes': head_classes,
        'medium_classes': medium_classes,
        'tail_classes': tail_classes,
        'class_counts': cls_num_list,
    })
    with open(os.path.join(args.output_dir, 'generation_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    elapsed = (time.time() - start) / 60
    print(f"Done! Generated {total_generated} images in {elapsed:.1f} min")


if __name__ == '__main__':
    main()
