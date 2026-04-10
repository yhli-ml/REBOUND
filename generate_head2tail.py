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
  # Basic usage with CIFAR-100-LT (one augmentation per image)
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./data/head2tail_cifar100_lt \\
      --top_k 3 --strength 0.6 --plan_mode per_image \\
      --per_image_scope medium_tail --aug_per_image 1

  # With LoRA fine-tuned model
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./data/head2tail_cifar100_lt \\
      --lora_weights ./lora_weights/cifar100_lt/final \\
      --plan_mode per_image --per_image_scope medium_tail --aug_per_image 1

  # Legacy target-mode ablation
  python generate_head2tail.py \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --output_dir ./data/head2tail_target_ablation \\
      --plan_mode target --target_num 500

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
import hashlib
import json
import os
import sys
import time
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from utils import get_class_split_lists


def parse_args():
    parser = argparse.ArgumentParser(
        description='Head2Tail augmentation for long-tail datasets')

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
    parser.add_argument('--plan_mode', type=str, default='per_image',
                        choices=['target', 'per_image'],
                        help='Augmentation plan mode. '
                             'per_image = canonical Head2Tail plan where every '
                             'image gets m augmentations. '
                             'target = legacy ablation mode that fills classes '
                             'up to target_num. '
                             '(head=self-aug, medium/tail=head2tail)')
    parser.add_argument('--aug_per_image', type=int, default=1,
                        help='Number of augmented images per original image '
                             '(only used in per_image plan mode)')
    parser.add_argument('--per_image_scope', type=str, default='medium_tail',
                        choices=['all', 'medium_tail', 'tail_only'],
                        help='Which classes receive per-image augmentation. '
                             'all = keep head self-augmentation plus medium/tail '
                             'Head2Tail transfer. '
                             'medium_tail = augment only medium/tail classes. '
                             'tail_only = augment only tail classes.')
    parser.add_argument('--head_strength', type=float, default=-1,
                        help='SDEdit strength for head class self-augmentation '
                             'in per_image mode. -1 = use same as --strength. '
                             'Typically smaller (e.g. 0.3-0.5) to preserve '
                             'head class features.')
    parser.add_argument('--target_num', type=int, default=-1,
                        help='Target samples per tail class after augmentation '
                             'in legacy target mode. '
                             '-1 = match head class median count')
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
                                head_threshold, tail_threshold,
                                dataset_name=None):
    """Compute how many images to generate for each tail class.

    Args:
        cls_num_list: Per-class training sample counts.
        mapping: Head-to-tail mapping from selector.
        target_num: Target count per class (-1 = median of head classes).
        max_aug_per_class: Maximum augmented images per class.
        augment_medium: Whether to include medium-shot classes.
        head_threshold: Head class threshold on non-CIFAR datasets.
        tail_threshold: Tail class threshold on non-CIFAR datasets.
        dataset_name: Dataset identifier for split policy.

    Returns:
        aug_plan: Dict[int, int] {class_idx: num_to_generate}
    """
    num_classes = len(cls_num_list)
    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )

    # Auto-determine target
    if target_num <= 0:
        head_counts = [cls_num_list[c] for c in head_classes]
        if head_counts:
            target_num = int(np.median(head_counts))
        else:
            target_num = max(cls_num_list) // 2
        print(f"[Plan] Auto target_num = {target_num} (median of head classes)")

    eligible_targets = set(tail_classes)
    if augment_medium:
        eligible_targets.update(medium_classes)

    aug_plan = {}
    for tail_c in mapping.keys():
        if tail_c not in eligible_targets:
            continue
        current = cls_num_list[tail_c]
        needed = max(0, target_num - current)
        if needed > 0:
            aug_plan[tail_c] = min(needed, max_aug_per_class)

    return aug_plan, target_num


def compute_per_image_plan(cls_num_list, num_classes, aug_per_image,
                          head_threshold, tail_threshold, per_image_scope,
                          dataset_name=None):
    """Compute per-image augmentation plan for ALL classes.

    Scope depends on `per_image_scope`.
    - all: every class gets m augmentations per image
    - medium_tail: only medium/tail classes are augmented
    - tail_only: only tail classes are augmented

    Args:
        cls_num_list: Per-class training sample counts.
        num_classes: Total number of classes.
        aug_per_image: Number of augmented images per original image (m).
        head_threshold: Head class threshold on non-CIFAR datasets.
        tail_threshold: Tail class threshold on non-CIFAR datasets.
        dataset_name: Dataset identifier for split policy.

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
    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )
    head_set = set(head_classes)
    medium_set = set(medium_classes)
    tail_set = set(tail_classes)

    for c in range(num_classes):
        n_current = cls_num_list[c]
        n_to_gen = n_current * aug_per_image
        if n_to_gen <= 0:
            continue

        if per_image_scope == 'tail_only':
            if c not in tail_set:
                continue
        elif per_image_scope == 'medium_tail':
            if c in head_set:
                continue

        aug_plan[c] = n_to_gen

        if per_image_scope == 'all' and c in head_set:
            head_plan[c] = n_to_gen
        else:
            tail_plan[c] = n_to_gen

    return aug_plan, head_plan, tail_plan


def get_random_mapping(num_classes, cls_num_list, head_threshold,
                        tail_threshold, top_k, dataset_name=None,
                        include_medium_targets=False):
    """Random head selection baseline (for ablation)."""
    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )
    target_classes = medium_classes + tail_classes if include_medium_targets else tail_classes

    mapping = {}
    for tc in target_classes:
        selected = random.sample(head_classes, min(top_k, len(head_classes)))
        mapping[tc] = [(hc, 0.0) for hc in selected]

    return mapping, head_classes, target_classes


def get_farthest_mapping(prototype_tensor, cls_num_list, head_threshold,
                          tail_threshold, top_k, dataset_name=None,
                          include_medium_targets=False):
    """Farthest head selection (for ablation)."""
    import torch.nn.functional as F

    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )
    target_classes = medium_classes + tail_classes if include_medium_targets else tail_classes

    prototypes = prototype_tensor.float()
    head_protos = prototypes[head_classes]
    tail_protos = prototypes[target_classes]

    sim_matrix = torch.mm(
        F.normalize(tail_protos, dim=1),
        F.normalize(head_protos, dim=1).t()
    )

    mapping = {}
    for t_idx, tc in enumerate(target_classes):
        sims = sim_matrix[t_idx]
        # Get BOTTOM k (farthest)
        _, bottom_idxs = torch.topk(sims, min(top_k, len(head_classes)),
                                     largest=False)
        mapping[tc] = [(head_classes[hi.item()], sims[hi].item())
                       for hi in bottom_idxs]

    return mapping, head_classes, target_classes


def compute_feature_prototypes(selector, feature_source, train_dataset):
    """Compute the prototype bank required by the current feature setting."""
    if feature_source == 'clip':
        return selector.compute_clip_prototypes()
    if feature_source == 'clip_image':
        return selector.compute_clip_image_prototypes(train_dataset)
    raise ValueError(f"Unsupported feature source: {feature_source}")


def build_nearest_source_cache(selector, class_samples, mapping):
    """Cache ranked source-sample lists for each target/source class pair."""
    cache = {}
    if selector.sample_features is None:
        return cache

    for tail_class, head_sources in mapping.items():
        for head_class, _ in head_sources:
            key = (tail_class, head_class)
            if key in cache:
                continue
            n_samples = len(class_samples.get(head_class, []))
            if n_samples <= 0:
                cache[key] = []
                continue
            cache[key] = selector.get_nearest_head_samples(
                tail_class, head_class, n_samples=n_samples
            )

    return cache


def select_head_source_sample(head_samples, tail_class, head_class, n_generated,
                              sample_selection, nearest_sample_cache,
                              nearest_sample_cursors):
    """Select a concrete source sample and expose its provenance metadata."""
    if sample_selection == 'random':
        sample_index = random.randrange(len(head_samples))
        return head_samples[sample_index], sample_index, None, 'random'

    if sample_selection == 'nearest':
        ranked_samples = nearest_sample_cache.get((tail_class, head_class), [])
        if ranked_samples:
            cursor = nearest_sample_cursors[(tail_class, head_class)]
            sample_rank = cursor % len(ranked_samples)
            sample_index = ranked_samples[sample_rank][0]
            nearest_sample_cursors[(tail_class, head_class)] += 1
            return (
                head_samples[sample_index],
                sample_index,
                sample_rank,
                'nearest',
            )

    sample_index = n_generated % len(head_samples)
    return head_samples[sample_index], sample_index, None, 'cycle'


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size)

    print("=" * 60)
    print("Head2Tail Augmentation (proposed method)")
    print("=" * 60)
    print(f"Dataset: {args.dataset} (imb_factor={args.imb_factor})")
    print(f"Strategy: {args.head_selection} head selection, "
          f"top_k={args.top_k}")
    print(f"Plan mode: {args.plan_mode}"
          + (f" (m={args.aug_per_image} per image)" if args.plan_mode == 'per_image' else ''))
    if args.plan_mode == 'per_image':
        print(f"Per-image scope: {args.per_image_scope}")
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

    # In per_image mode and in target mode with --augment_medium, medium-shot
    # classes also need head-to-tail mappings, so we widen the target set.
    include_medium_targets = (args.plan_mode == 'per_image' or args.augment_medium)

    if args.head_selection == 'nearest':
        compute_feature_prototypes(selector, args.feature_source, raw_ds)

        mapping, head_classes, tail_classes = selector.get_head2tail_mapping(
            cls_num_list, top_k=args.top_k,
            head_threshold=args.head_threshold,
            tail_threshold=args.tail_threshold,
            include_medium_targets=include_medium_targets,
        )

    elif args.head_selection == 'random':
        mapping, head_classes, tail_classes = get_random_mapping(
            num_classes, cls_num_list, args.head_threshold,
            args.tail_threshold, args.top_k,
            dataset_name=args.dataset,
            include_medium_targets=include_medium_targets,
        )
        if args.sample_selection == 'nearest' and args.feature_source == 'clip_image':
            compute_feature_prototypes(selector, args.feature_source, raw_ds)
        else:
            selector.compute_clip_prototypes()

    elif args.head_selection == 'farthest':
        protos = compute_feature_prototypes(selector, args.feature_source, raw_ds)
        mapping, head_classes, tail_classes = get_farthest_mapping(
            protos, cls_num_list, args.head_threshold,
            args.tail_threshold, args.top_k,
            dataset_name=args.dataset,
            include_medium_targets=include_medium_targets,
        )

    elif args.head_selection == 'all':
        if args.sample_selection == 'nearest' and args.feature_source == 'clip_image':
            compute_feature_prototypes(selector, args.feature_source, raw_ds)
        else:
            selector.compute_clip_prototypes()
        # Use all head classes for each tail class
        head_classes, medium_classes, tail_classes = get_class_split_lists(
            cls_num_list,
            dataset_name=args.dataset,
            many_thr=args.head_threshold,
            few_thr=args.tail_threshold,
        )
        if include_medium_targets:
            tail_classes = medium_classes + tail_classes
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
            args.head_threshold, args.tail_threshold, args.per_image_scope,
            dataset_name=args.dataset,
        )
        target_num = -1  # not applicable

        total_to_generate = sum(aug_plan.values())
        print(f"  Plan mode: per_image (m={args.aug_per_image}, "
              f"scope={args.per_image_scope})")
        print(f"  Head classes to self-augment: {len(head_aug_plan)} "
              f"({sum(head_aug_plan.values())} images)")
        print(f"  Medium/tail classes (head2tail): {len(tail_aug_plan)} "
              f"({sum(tail_aug_plan.values())} images)")
        print(f"  Total images to generate: {total_to_generate}")
    else:
        # Target mode: fill tail classes up to target_num
        aug_plan, target_num = compute_augmentation_plan(
            cls_num_list, mapping, args.target_num, args.max_aug_per_class,
            args.augment_medium, args.head_threshold, args.tail_threshold,
            dataset_name=args.dataset,
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

    nearest_sample_cache = {}
    nearest_sample_cursors = defaultdict(int)
    if args.sample_selection == 'nearest':
        if selector.sample_features is not None:
            nearest_sample_cache = build_nearest_source_cache(
                selector, class_samples, mapping
            )
            print(f"  Cached nearest source samples for "
                  f"{len(nearest_sample_cache)} target/source pairs")
        else:
            print("  [WARN] sample_selection=nearest requires image features; "
                  "falling back to deterministic source cycling.")

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

                    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
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
                        'source_sample_index': img_idx,
                        'source_sample_rank': 0,
                        'source_sample_strategy': 'self',
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

                sample, source_sample_index, source_sample_rank, source_sample_strategy = \
                    select_head_source_sample(
                        head_samples=head_samples,
                        tail_class=tail_c,
                        head_class=head_c,
                        n_generated=n_generated,
                        sample_selection=args.sample_selection,
                        nearest_sample_cache=nearest_sample_cache,
                        nearest_sample_cursors=nearest_sample_cursors,
                    )

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
                prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
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
                    'source_sample_index': source_sample_index,
                    'source_sample_rank': source_sample_rank,
                    'source_sample_strategy': source_sample_strategy,
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
        'method': 'head2tail',
        'dataset': args.dataset,
        'imb_factor': args.imb_factor,
        'plan_mode': args.plan_mode,
        'per_image_scope': args.per_image_scope if args.plan_mode == 'per_image' else None,
        'head_selection': args.head_selection,
        'sample_selection': args.sample_selection,
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
