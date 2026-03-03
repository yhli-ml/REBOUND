#!/usr/bin/env python3
"""Offline Head-to-Tail augmentation for long-tail datasets.

Complete pipeline:
  1. Load long-tail dataset and identify head/tail class split
  2. Compute feature-space prototypes (CLIP-based)
  3. For each tail class, find K nearest head classes
  4. Sample images from nearest head classes
  5. Use SDEdit (img2img) to transfer head images → tail class semantics
  6. Save generated images in DiffuseMix-compatible format

The generated images can be loaded by the existing DiffuseMixDataset
wrapper for downstream classifier training.

Usage:
  # Basic usage with CIFAR-100-LT
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./data/head2tail_cifar100_lt \\
      --target_num 500 --top_k 3 --strength 0.6

  # With LoRA fine-tuned model
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./data/head2tail_cifar100_lt \\
      --lora_weights ./lora_weights/cifar100_lt/final \\
      --target_num 500

  # With custom prompts
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data \\
      --custom_prompts ./configs/cifar100_prompts.json \\
      --output_dir ./data/head2tail_cifar100_lt

Output structure (compatible with DiffuseMixDataset):
  output_dir/
    metadata.json
    augmented_list.txt
    head2tail_mapping.json   # Head→Tail mapping for analysis
    class_XXXX/
      h2t_00000_from{head_cls}_{prompt_hash}.jpg
      ...
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Head-to-Tail augmentation for long-tail datasets')

    # ===== Dataset =====
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt',
                                 'inaturalist', 'places_lt'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--imb_factor', type=float, default=0.01)

    # ===== Head-to-Tail Strategy =====
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of nearest head classes to use as source')
    parser.add_argument('--head_threshold', type=int, default=100,
                        help='Classes with > this count are "head"')
    parser.add_argument('--tail_threshold', type=int, default=20,
                        help='Classes with <= this count are "tail"')
    parser.add_argument('--feature_source', type=str, default='clip_image',
                        choices=['clip', 'clip_image'],
                        help='Feature source: clip (text) or clip_image (image)')
    parser.add_argument('--head_selection', type=str, default='nearest',
                        choices=['nearest', 'random', 'farthest', 'all'],
                        help='Head class selection strategy (for ablation)')
    parser.add_argument('--sample_selection', type=str, default='nearest',
                        choices=['nearest', 'random'],
                        help='How to select specific samples from head class. '
                             'nearest = samples closest to tail prototype, '
                             'random = random sampling')

    # ===== Generation =====
    parser.add_argument('--model_id', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='Base Stable Diffusion model')
    parser.add_argument('--pipeline_type', type=str, default='img2img',
                        choices=['img2img', 'pix2pix'],
                        help='Generation pipeline type')
    parser.add_argument('--lora_weights', type=str, default='',
                        help='Path to LoRA weights (optional)')
    parser.add_argument('--strength', type=float, default=0.6,
                        help='SDEdit strength (0=no change, 1=full regen)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--gen_size', type=int, default=512,
                        help='Generation resolution')
    parser.add_argument('--n_prompts_per_class', type=int, default=5,
                        help='Number of diverse prompts per tail class')
    parser.add_argument('--custom_prompts', type=str, default='',
                        help='Path to custom prompts JSON file')

    # ===== Augmentation Plan =====
    parser.add_argument('--plan_mode', type=str, default='target',
                        choices=['target', 'per_image'],
                        help='Augmentation plan mode. '
                             'target = fill tail classes up to target_num. '
                             'per_image = augment every image m times '
                             '(head=self-aug, medium/tail=head2tail)')
    parser.add_argument('--aug_per_image', type=int, default=1,
                        help='Number of augmented images per original image '
                             '(only used in per_image plan mode)')
    parser.add_argument('--head_strength', type=float, default=-1,
                        help='SDEdit strength for head class self-augmentation '
                             'in per_image mode. -1 = use same as --strength. '
                             'Typically smaller (e.g. 0.3-0.5) to preserve '
                             'head class features.')
    parser.add_argument('--target_num', type=int, default=-1,
                        help='Target samples per tail class after augmentation. '
                             '-1 = match head class median count '
                             '(only used in target plan mode)')
    parser.add_argument('--max_aug_per_class', type=int, default=500,
                        help='Max augmented images per class '
                             '(only used in target plan mode)')
    parser.add_argument('--augment_medium', action='store_true',
                        help='Also augment medium-shot classes '
                             '(only used in target plan mode)')

    # ===== Output =====
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_size', type=int, default=0,
                        help='Output save size. 0 = same as gen_size. '
                             'For CIFAR use 32.')

    # ===== System =====
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.save_size == 0:
        is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')
        args.save_size = 32 if is_cifar else args.gen_size

    return args


def load_dataset_samples(args):
    """Load dataset and return organized samples.

    Returns:
        class_samples: Dict[int, List[PIL.Image or np.ndarray]]
        cls_num_list: List[int]
        num_classes: int
        is_cifar: bool
    """
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

        cls_num_list = ds.cls_num_list

        # Organize by class
        class_samples = {c: [] for c in range(num_classes)}
        for i in range(len(ds.data)):
            class_samples[ds.targets[i]].append(ds.data[i])  # numpy arrays

        return class_samples, cls_num_list, num_classes, True, ds

    else:
        from datasets import get_dataset
        img_root = args.img_root if args.img_root else None
        ds = get_dataset(args.dataset, args.data_root, train=True,
                         transform=None, img_root=img_root)
        cls_num_list = ds.cls_num_list

        if args.dataset == 'imagenet_lt':
            num_classes = 1000
        elif args.dataset == 'inaturalist':
            num_classes = 8142
        else:
            num_classes = 365

        class_samples = {c: [] for c in range(num_classes)}
        for path, label in zip(ds.img_paths, ds.targets):
            class_samples[label].append(path)  # file paths

        return class_samples, cls_num_list, num_classes, False, ds


def compute_augmentation_plan(cls_num_list, mapping, target_num,
                                max_aug_per_class, augment_medium,
                                head_threshold, tail_threshold):
    """Compute how many images to generate for each tail class.

    Args:
        cls_num_list: Per-class training sample counts.
        mapping: Head-to-tail mapping from selector.
        target_num: Target count per class (-1 = median of head classes).
        max_aug_per_class: Maximum augmented images per class.
        augment_medium: Whether to include medium-shot classes.
        head_threshold: Head class threshold.
        tail_threshold: Tail class threshold.

    Returns:
        aug_plan: Dict[int, int] {class_idx: num_to_generate}
    """
    num_classes = len(cls_num_list)

    # Auto-determine target
    if target_num <= 0:
        head_counts = [cls_num_list[c] for c in range(num_classes)
                       if cls_num_list[c] > head_threshold]
        if head_counts:
            target_num = int(np.median(head_counts))
        else:
            target_num = max(cls_num_list) // 2
        print(f"[Plan] Auto target_num = {target_num} (median of head classes)")

    aug_plan = {}
    for tail_c in mapping.keys():
        current = cls_num_list[tail_c]
        # Include medium classes if requested
        if not augment_medium and current > tail_threshold:
            continue
        needed = max(0, target_num - current)
        if needed > 0:
            aug_plan[tail_c] = min(needed, max_aug_per_class)

    return aug_plan, target_num


def compute_per_image_plan(cls_num_list, num_classes, aug_per_image,
                          head_threshold, tail_threshold):
    """Compute per-image augmentation plan for ALL classes.

    Every class is augmented: each original image produces m new images.
    - Head classes: self-augmentation (same class image + same class prompt)
    - Medium/tail classes: head-to-tail transfer (unchanged approach)

    Args:
        cls_num_list: Per-class training sample counts.
        num_classes: Total number of classes.
        aug_per_image: Number of augmented images per original image (m).
        head_threshold: Head class threshold.
        tail_threshold: Tail class threshold.

    Returns:
        aug_plan: Dict[int, int] {class_idx: num_to_generate}
        head_plan: Dict[int, int] {head_class_idx: num_to_generate}
            (head classes needing self-augmentation)
        tail_plan: Dict[int, int] {tail/medium_class_idx: num_to_generate}
            (medium/tail classes needing head-to-tail transfer)
    """
    aug_plan = {}      # combined plan for all classes
    head_plan = {}     # head class self-augmentation
    tail_plan = {}     # medium/tail head-to-tail transfer

    for c in range(num_classes):
        n_current = cls_num_list[c]
        n_to_gen = n_current * aug_per_image
        if n_to_gen <= 0:
            continue

        aug_plan[c] = n_to_gen

        if n_current > head_threshold:
            head_plan[c] = n_to_gen
        else:
            tail_plan[c] = n_to_gen

    return aug_plan, head_plan, tail_plan


def get_random_mapping(num_classes, cls_num_list, head_threshold,
                        tail_threshold, top_k):
    """Random head selection baseline (for ablation)."""
    head_classes = [c for c in range(num_classes)
                    if cls_num_list[c] > head_threshold]
    tail_classes = [c for c in range(num_classes)
                    if cls_num_list[c] <= tail_threshold]

    mapping = {}
    for tc in tail_classes:
        selected = random.sample(head_classes, min(top_k, len(head_classes)))
        mapping[tc] = [(hc, 0.0) for hc in selected]

    return mapping, head_classes, tail_classes


def get_farthest_mapping(prototype_tensor, cls_num_list, head_threshold,
                          tail_threshold, top_k):
    """Farthest head selection (for ablation)."""
    import torch.nn.functional as F

    num_classes = len(cls_num_list)
    head_classes = [c for c in range(num_classes)
                    if cls_num_list[c] > head_threshold]
    tail_classes = [c for c in range(num_classes)
                    if cls_num_list[c] <= tail_threshold]

    prototypes = prototype_tensor.float()
    head_protos = prototypes[head_classes]
    tail_protos = prototypes[tail_classes]

    sim_matrix = torch.mm(
        F.normalize(tail_protos, dim=1),
        F.normalize(head_protos, dim=1).t()
    )

    mapping = {}
    for t_idx, tc in enumerate(tail_classes):
        sims = sim_matrix[t_idx]
        # Get BOTTOM k (farthest)
        _, bottom_idxs = torch.topk(sims, min(top_k, len(head_classes)),
                                     largest=False)
        mapping[tc] = [(head_classes[hi.item()], sims[hi].item())
                       for hi in bottom_idxs]

    return mapping, head_classes, tail_classes


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size)

    print("=" * 60)
    print("Head-to-Tail Transfer Augmentation")
    print("=" * 60)
    print(f"Dataset: {args.dataset} (imb_factor={args.imb_factor})")
    print(f"Strategy: {args.head_selection} head selection, "
          f"top_k={args.top_k}")
    print(f"Plan mode: {args.plan_mode}"
          + (f" (m={args.aug_per_image} per image)" if args.plan_mode == 'per_image' else ''))
    print(f"SDEdit: strength={args.strength}, "
          f"guidance_scale={args.guidance_scale}")
    if args.plan_mode == 'per_image' and args.head_strength >= 0:
        print(f"Head self-aug strength: {args.head_strength}")
    print(f"Pipeline: {args.pipeline_type}, model={args.model_id}")
    if args.lora_weights:
        print(f"LoRA weights: {args.lora_weights}")
    print(f"Output: {args.output_dir}")
    print()

    # Resolve head_strength
    head_strength = args.head_strength if args.head_strength >= 0 else args.strength

    # ===== Step 1: Load dataset =====
    print("[1/5] Loading dataset...")
    class_samples, cls_num_list, num_classes, is_cifar, raw_ds = \
        load_dataset_samples(args)

    print(f"  Classes: {num_classes}, Total samples: {sum(cls_num_list)}")
    print(f"  Max count: {max(cls_num_list)}, Min: {min(cls_num_list)}")

    # ===== Step 2: Compute prototypes & head-tail mapping =====
    print("\n[2/5] Computing feature-space prototypes...")
    from augment.head2tail_selector import HeadClassSelector

    selector = HeadClassSelector(
        dataset_name=args.dataset, device=args.device,
        feature_source=args.feature_source
    )

    # In per_image mode, medium classes also need head-to-tail mapping,
    # so use head_threshold as the effective tail_threshold for mapping.
    effective_tail_threshold = (
        args.head_threshold if args.plan_mode == 'per_image'
        else args.tail_threshold
    )

    if args.head_selection == 'nearest':
        if args.feature_source == 'clip':
            selector.compute_clip_prototypes()
        elif args.feature_source == 'clip_image':
            selector.compute_clip_image_prototypes(raw_ds)

        mapping, head_classes, tail_classes = selector.get_head2tail_mapping(
            cls_num_list, top_k=args.top_k,
            head_threshold=args.head_threshold,
            tail_threshold=effective_tail_threshold
        )

    elif args.head_selection == 'random':
        mapping, head_classes, tail_classes = get_random_mapping(
            num_classes, cls_num_list, args.head_threshold,
            effective_tail_threshold, args.top_k
        )
        # Still compute prototypes for logging
        selector.compute_clip_prototypes()

    elif args.head_selection == 'farthest':
        protos = selector.compute_clip_prototypes()
        mapping, head_classes, tail_classes = get_farthest_mapping(
            protos, cls_num_list, args.head_threshold,
            effective_tail_threshold, args.top_k
        )

    elif args.head_selection == 'all':
        selector.compute_clip_prototypes()
        # Use all head classes for each tail class
        head_classes = [c for c in range(num_classes)
                        if cls_num_list[c] > args.head_threshold]
        tail_classes = [c for c in range(num_classes)
                        if cls_num_list[c] <= effective_tail_threshold]
        mapping = {tc: [(hc, 0.0) for hc in head_classes] for tc in tail_classes}

    # Save mapping for analysis
    os.makedirs(args.output_dir, exist_ok=True)
    mapping_path = os.path.join(args.output_dir, 'head2tail_mapping.json')
    selector.save_mapping(mapping, head_classes, tail_classes,
                           cls_num_list, mapping_path)

    # Print mapping summary
    from augment.head2tail_prompts import get_class_names
    class_names = get_class_names(args.dataset)
    print(f"\n  Head-to-Tail mapping (top {args.top_k}):")
    for tc in sorted(list(mapping.keys()))[:10]:  # Show first 10
        tail_name = class_names[tc] if tc < len(class_names) else f"cls_{tc}"
        sources = ", ".join([
            f"{class_names[hc]}({sim:.3f})" if hc < len(class_names) else f"cls_{hc}({sim:.3f})"
            for hc, sim in mapping[tc]
        ])
        print(f"    {tail_name} (n={cls_num_list[tc]}) ← {sources}")
    if len(mapping) > 10:
        print(f"    ... ({len(mapping)} tail classes total)")

    # ===== Step 3: Compute augmentation plan =====
    print("\n[3/5] Computing augmentation plan...")

    if args.plan_mode == 'per_image':
        # Per-image mode: every class gets m augmentations per image
        aug_plan, head_aug_plan, tail_aug_plan = compute_per_image_plan(
            cls_num_list, num_classes, args.aug_per_image,
            args.head_threshold, args.tail_threshold
        )
        target_num = -1  # not applicable

        total_to_generate = sum(aug_plan.values())
        print(f"  Plan mode: per_image (m={args.aug_per_image})")
        print(f"  Head classes to self-augment: {len(head_aug_plan)} "
              f"({sum(head_aug_plan.values())} images)")
        print(f"  Medium/tail classes (head2tail): {len(tail_aug_plan)} "
              f"({sum(tail_aug_plan.values())} images)")
        print(f"  Total images to generate: {total_to_generate}")
    else:
        # Target mode: fill tail classes up to target_num
        aug_plan, target_num = compute_augmentation_plan(
            cls_num_list, mapping, args.target_num, args.max_aug_per_class,
            args.augment_medium, args.head_threshold, args.tail_threshold
        )
        head_aug_plan = {}  # no head self-augmentation in target mode
        tail_aug_plan = aug_plan

        total_to_generate = sum(aug_plan.values())
        print(f"  Target per class: {target_num}")
        print(f"  Classes to augment: {len(aug_plan)}")
        print(f"  Total images to generate: {total_to_generate}")

    if total_to_generate == 0:
        print("Nothing to generate. Exiting.")
        return

    # ===== Step 4: Prepare prompts =====
    print("\n[4/5] Preparing prompts...")
    if args.custom_prompts and os.path.exists(args.custom_prompts):
        from augment.head2tail_prompts import load_custom_prompts
        all_prompts = load_custom_prompts(args.custom_prompts)
        print(f"  Loaded custom prompts from {args.custom_prompts}")
    else:
        from augment.head2tail_prompts import get_all_prompts
        all_prompts = get_all_prompts(
            args.dataset, n_prompts_per_class=args.n_prompts_per_class,
            seed=args.seed
        )
        print(f"  Generated {args.n_prompts_per_class} prompts per class")

    # Show sample prompts
    sample_tail = list(aug_plan.keys())[0]
    print(f"  Example prompts for class {sample_tail} "
          f"({class_names[sample_tail] if sample_tail < len(class_names) else sample_tail}):")
    for p in all_prompts.get(sample_tail, ["a photo"])[:3]:
        print(f"    - {p}")

    # ===== Step 5: Generate images =====
    print(f"\n[5/5] Generating augmented images...")
    from augment.head2tail_generator import Head2TailGenerator

    generator = Head2TailGenerator(
        model_id=args.model_id,
        pipeline_type=args.pipeline_type,
        device=args.device,
        lora_weights=args.lora_weights if args.lora_weights else None,
    )

    metadata = {}
    global_count = 0
    start_time = time.time()

    # --- 5a: Head class self-augmentation (per_image mode only) ---
    if head_aug_plan:
        print(f"\n  === Head class self-augmentation ===")
        for head_c in sorted(head_aug_plan.keys()):
            class_dir = os.path.join(args.output_dir, f'class_{head_c:04d}')
            os.makedirs(class_dir, exist_ok=True)

            class_key = f'class_{head_c:04d}'
            metadata[class_key] = []

            head_name = class_names[head_c] if head_c < len(class_names) else f"class_{head_c}"
            prompts = all_prompts.get(head_c, [f"a photo of a {head_name.replace('_', ' ')}"])
            own_samples = class_samples.get(head_c, [])

            if not own_samples:
                print(f"  [SKIP] Head class {head_c} ({head_name}): no samples")
                continue

            n_to_gen = head_aug_plan[head_c]
            print(f"  Head {head_c:4d} ({head_name}): self-aug {len(own_samples)} "
                  f"images x {args.aug_per_image} = {n_to_gen}")

            n_generated = 0
            for img_idx, sample in enumerate(own_samples):
                # Convert to PIL
                if isinstance(sample, np.ndarray):
                    src_pil = Image.fromarray(sample).convert('RGB')
                elif isinstance(sample, str):
                    src_pil = Image.open(sample).convert('RGB')
                else:
                    src_pil = sample.convert('RGB')

                for aug_idx in range(args.aug_per_image):
                    # Use diverse prompts by cycling
                    prompt = prompts[(img_idx * args.aug_per_image + aug_idx) % len(prompts)]

                    try:
                        gen_img = generator.generate_with_retry(
                            src_pil, prompt,
                            strength=head_strength,
                            guidance_scale=args.guidance_scale,
                            size=gen_size,
                            num_inference_steps=args.num_inference_steps,
                            max_retries=2,
                        )
                    except Exception as e:
                        print(f"    [WARN] Self-aug failed for class {head_c} "
                              f"img {img_idx} aug {aug_idx}: {e}")
                        continue

                    if gen_img is None:
                        continue

                    if save_size != gen_size:
                        gen_img = gen_img.resize(save_size, Image.LANCZOS)

                    prompt_hash = abs(hash(prompt)) % 10000
                    fname = (f'self_{img_idx:05d}_aug{aug_idx}'
                             f'_{prompt_hash}.jpg')
                    fpath = os.path.join(class_dir, fname)
                    gen_img.save(fpath, quality=95)

                    metadata[class_key].append({
                        'path': os.path.join(class_key, fname),
                        'label': head_c,
                        'source_class': head_c,
                        'source_name': head_name,
                        'similarity': 1.0,
                        'prompt': prompt,
                        'strength': head_strength,
                        'aug_type': 'self',
                    })

                    n_generated += 1
                    global_count += 1

                    if global_count % 50 == 0:
                        elapsed = time.time() - start_time
                        speed = global_count / elapsed
                        remaining = (total_to_generate - global_count) / max(speed, 1e-6)
                        print(f"    Progress: {global_count}/{total_to_generate} "
                              f"({speed:.1f} img/s, ETA: {remaining / 60:.1f} min)")

            if n_generated < n_to_gen:
                print(f"    [WARN] Only generated {n_generated}/{n_to_gen} "
                      f"for head class {head_c}")

    # --- 5b: Medium/tail class head-to-tail transfer ---
    if tail_aug_plan:
        print(f"\n  === Medium/tail class head-to-tail transfer ===")
        for tail_c in sorted(tail_aug_plan.keys()):
            n_to_gen = tail_aug_plan[tail_c]
            class_dir = os.path.join(args.output_dir, f'class_{tail_c:04d}')
            os.makedirs(class_dir, exist_ok=True)

            class_key = f'class_{tail_c:04d}'
            if class_key not in metadata:
                metadata[class_key] = []

            tail_name = class_names[tail_c] if tail_c < len(class_names) else f"class_{tail_c}"
            head_sources = mapping.get(tail_c, [])
            prompts = all_prompts.get(tail_c, [f"a photo of a {tail_name.replace('_', ' ')}"])

            if not head_sources:
                print(f"  [SKIP] Class {tail_c} ({tail_name}): no head sources")
                continue

            print(f"  Class {tail_c:4d} ({tail_name}): generating {n_to_gen} "
                  f"from {len(head_sources)} head classes...")

            n_generated = 0
            attempt_count = 0
            max_attempts = n_to_gen * 3  # Safety limit

            while n_generated < n_to_gen and attempt_count < max_attempts:
                attempt_count += 1

                # Pick a head class source (cycle through nearest ones)
                source_idx = n_generated % len(head_sources)
                head_c, similarity = head_sources[source_idx]

                # Pick a head class sample
                head_samples = class_samples.get(head_c, [])
                if not head_samples:
                    continue

                if args.sample_selection == 'random':
                    sample = random.choice(head_samples)
                else:
                    # Cycle through samples
                    sample = head_samples[n_generated % len(head_samples)]

                # Convert to PIL
                if isinstance(sample, np.ndarray):
                    head_pil = Image.fromarray(sample).convert('RGB')
                elif isinstance(sample, str):
                    head_pil = Image.open(sample).convert('RGB')
                else:
                    head_pil = sample.convert('RGB')

                # Pick a prompt
                prompt = prompts[n_generated % len(prompts)]

                # Generate
                try:
                    gen_img = generator.generate_with_retry(
                        head_pil, prompt,
                        strength=args.strength,
                        guidance_scale=args.guidance_scale,
                        size=gen_size,
                        num_inference_steps=args.num_inference_steps,
                        max_retries=2,
                    )
                except Exception as e:
                    print(f"    [WARN] Generation failed: {e}")
                    continue

                if gen_img is None:
                    continue

                # Resize and save
                if save_size != gen_size:
                    gen_img = gen_img.resize(save_size, Image.LANCZOS)

                head_name = class_names[head_c] if head_c < len(class_names) else f"cls{head_c}"
                prompt_hash = abs(hash(prompt)) % 10000
                fname = f'h2t_{n_generated:05d}_from{head_c}_{head_name}_{prompt_hash}.jpg'
                fpath = os.path.join(class_dir, fname)
                gen_img.save(fpath, quality=95)

                metadata[class_key].append({
                    'path': os.path.join(class_key, fname),
                    'label': tail_c,
                    'source_class': head_c,
                    'source_name': head_name,
                    'similarity': round(similarity, 4),
                    'prompt': prompt,
                    'strength': args.strength,
                    'aug_type': 'head2tail',
                })

                n_generated += 1
                global_count += 1

                if global_count % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = global_count / elapsed
                    remaining = (total_to_generate - global_count) / max(speed, 1e-6)
                    print(f"    Progress: {global_count}/{total_to_generate} "
                          f"({speed:.1f} img/s, ETA: {remaining / 60:.1f} min)")

            if n_generated < n_to_gen:
                print(f"    [WARN] Only generated {n_generated}/{n_to_gen} "
                      f"for class {tail_c}")

    # ===== Save metadata =====
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    txt_path = os.path.join(args.output_dir, 'augmented_list.txt')
    with open(txt_path, 'w') as f:
        for class_key in sorted(metadata.keys()):
            for entry in metadata[class_key]:
                f.write(f"{entry['path']} {entry['label']}\n")

    # Save generation config
    config = {
        'dataset': args.dataset,
        'imb_factor': args.imb_factor,
        'plan_mode': args.plan_mode,
        'head_selection': args.head_selection,
        'top_k': args.top_k,
        'strength': args.strength,
        'head_strength': head_strength,
        'guidance_scale': args.guidance_scale,
        'model_id': args.model_id,
        'pipeline_type': args.pipeline_type,
        'lora_weights': args.lora_weights,
        'target_num': target_num,
        'aug_per_image': args.aug_per_image if args.plan_mode == 'per_image' else None,
        'total_generated': global_count,
        'gen_size': args.gen_size,
        'save_size': args.save_size,
        'seed': args.seed,
    }
    config_path = os.path.join(args.output_dir, 'generation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {global_count} images in {elapsed / 60:.1f} min")
    print(f"Metadata: {meta_path}")
    print(f"Image list: {txt_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
