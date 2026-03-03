#!/usr/bin/env python3
"""Offline DiffuseMix augmentation for long-tail datasets.

Generates augmented images for minority (tail) classes using the DiffuseMix
pipeline (InstructPix2Pix style transfer + concatenation + fractal blending).

Key features for long-tail learning:
  - Automatically detects the per-class distribution.
  - Generates MORE augmented samples for classes with FEWER training examples.
  - Supports CIFAR-10/100-LT (extracts images from numpy arrays) and
    ImageNet-LT (reads from disk).

Usage examples:
  # CIFAR-100-LT: augment tail classes to have at least 200 images each
  python generate_diffusemix.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --fractal_dir /path/to/deviantart \\
      --prompts "sunset,Autumn,watercolor art" \\
      --target_num 200 --output_dir ./data/diffusemix_cifar100_lt \\
      --gen_size 256

  # ImageNet-LT: augment classes with fewer than 100 images
  python generate_diffusemix.py \\
      --dataset imagenet_lt \\
      --data_root ./data/ImageNet_LT \\
      --img_root /path/to/imagenet \\
      --fractal_dir /path/to/deviantart \\
      --prompts "sunset,Autumn" \\
      --target_num 100 --output_dir ./data/diffusemix_imagenet_lt

Output structure:
  output_dir/
    metadata.json          # {class_label: [{"path": "...", "label": int}, ...]}
    class_000/
      aug_00000_sunset.jpg
      aug_00001_Autumn.jpg
      ...
    class_001/
      ...
"""

import argparse
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
        description='Offline DiffuseMix augmentation for long-tail datasets')

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

    # Long-tail strategy
    parser.add_argument('--target_num', type=int, default=-1,
                        help='Target number of samples per class after augmentation. '
                             '-1 = augment all classes equally')
    parser.add_argument('--max_aug_per_class', type=int, default=500,
                        help='Maximum augmented images to generate per class')
    parser.add_argument('--min_class_count', type=int, default=0,
                        help='Only augment classes with fewer than this many samples. '
                             '0 = auto (use target_num)')

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


def compute_augmentation_plan(cls_num_list, target_num, max_aug_per_class,
                               min_class_count, prompts):
    """Compute how many augmented images to generate per class.

    Args:
        cls_num_list: List of per-class sample counts.
        target_num: Target total per class. -1 = augment all equally.
        max_aug_per_class: Max augmented images per class.
        min_class_count: Only augment classes with count < this.
        prompts: List of prompts (affects multiplier).

    Returns:
        aug_plan: dict {class_idx: num_images_to_generate}
    """
    num_classes = len(cls_num_list)
    num_prompts = len(prompts)
    aug_plan = {}

    if target_num <= 0:
        # Augment all classes: generate `num_prompts` images per original
        for c in range(num_classes):
            aug_plan[c] = min(cls_num_list[c] * num_prompts, max_aug_per_class)
    else:
        # Only augment classes below target
        for c in range(num_classes):
            current = cls_num_list[c]
            if min_class_count > 0 and current >= min_class_count:
                continue
            if current >= target_num:
                continue
            needed = target_num - current
            aug_plan[c] = min(needed, max_aug_per_class)

    return aug_plan


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = [p.strip() for p in args.prompts.split(',')]
    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size) if args.save_size > 0 else gen_size

    print("=" * 60)
    print("DiffuseMix Offline Augmentation for Long-Tail Learning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Prompts: {prompts}")
    print(f"Target num per class: {args.target_num}")
    print(f"Output: {args.output_dir}")
    print()

    # Load dataset
    print("[1/4] Loading dataset...")
    samples, cls_num_list, num_classes, is_cifar = get_dataset_samples(args)
    print(f"  Classes: {num_classes}, Total samples: {len(samples)}")
    print(f"  Max class count: {max(cls_num_list)}, Min: {min(cls_num_list)}")

    # Organize samples by class
    class_samples = {c: [] for c in range(num_classes)}
    for item, label in samples:
        class_samples[label].append(item)

    # Compute augmentation plan
    aug_plan = compute_augmentation_plan(
        cls_num_list, args.target_num, args.max_aug_per_class,
        args.min_class_count, prompts)

    total_to_generate = sum(aug_plan.values())
    classes_to_augment = len(aug_plan)
    print(f"\n[2/4] Augmentation plan:")
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
    print("[3/4] Loading fractal images...")
    from augment.diffusemix_utils import load_fractal_images
    fractal_imgs = load_fractal_images(args.fractal_dir, size=gen_size)

    # Load diffusion model
    print("[3/4] Loading InstructPix2Pix model...")
    from augment.diffusemix_handler import ModelHandler
    model = ModelHandler(model_id=args.model_id, device=args.device)

    from augment.diffusemix_utils import (
        combine_images, blend_with_fractal, is_black_image
    )

    # Generate augmented images
    print(f"\n[4/4] Generating augmented images...")
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
        n_generated = 0

        print(f"  Class {class_idx:4d}: {n_original} original, "
              f"generating {n_to_gen} augmented...")

        # Cycle through originals, varying prompts
        sample_idx = 0
        while n_generated < n_to_gen:
            # Pick an original image (cycle)
            item = class_original[sample_idx % n_original]
            sample_idx += 1

            # Pick a prompt (random)
            prompt = random.choice(prompts)

            # Prepare original as PIL
            if is_cifar:
                # item is numpy array (H, W, C) - uint8
                original_pil = Image.fromarray(item).convert('RGB')
                original_resized = original_pil.resize(gen_size)
            else:
                # item is file path
                original_pil = Image.open(item).convert('RGB')
                original_resized = original_pil.resize(gen_size)

            # Step 1: Style transfer
            try:
                gen_imgs = model.generate_images_from_pil(
                    prompt, original_resized, num_images=1,
                    guidance_scale=args.guidance_scale, size=gen_size)
            except Exception as e:
                print(f"    [WARN] Generation failed for class {class_idx}: {e}")
                continue

            for gen_img in gen_imgs:
                if n_generated >= n_to_gen:
                    break

                gen_img = gen_img.resize(gen_size)
                if is_black_image(gen_img):
                    continue

                # Step 2: Concatenate
                combined = combine_images(original_resized, gen_img)

                # Step 3: Fractal blend
                fractal = random.choice(fractal_imgs)
                blended = blend_with_fractal(combined, fractal, args.fractal_alpha)

                # Save
                if save_size != gen_size:
                    blended = blended.resize(save_size)

                fname = f'aug_{n_generated:05d}_{prompt.replace(" ", "_")}.jpg'
                fpath = os.path.join(class_dir, fname)
                blended.save(fpath, quality=95)

                metadata[class_key].append({
                    'path': os.path.join(class_key, fname),
                    'label': class_idx,
                })
                n_generated += 1
                global_count += 1

                if global_count % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = global_count / elapsed
                    remaining = (total_to_generate - global_count) / max(speed, 1e-6)
                    print(f"    Progress: {global_count}/{total_to_generate} "
                          f"({speed:.1f} img/s, ETA: {remaining/60:.1f} min)")

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

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {global_count} augmented images in {elapsed/60:.1f} min")
    print(f"Metadata: {meta_path}")
    print(f"Image list: {txt_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
