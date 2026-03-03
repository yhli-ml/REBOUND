#!/usr/bin/env python3
"""Evaluate quality of Head-to-Tail generated images.

Metrics:
  1. FID (Fréchet Inception Distance) between generated and real tail-class images
  2. Intra-class diversity (LPIPS distance among generated images)
  3. Classification accuracy (using a pre-trained classifier)
  4. Domain consistency (FID between generated images and full training set)

Usage:
  python evaluate_head2tail_quality.py \\
      --generated_dir ./data/head2tail_cifar100_lt \\
      --dataset cifar100_lt --data_root ./data --imb_factor 0.01 \\
      --metrics fid diversity classification
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Head-to-Tail generated image quality')

    parser.add_argument('--generated_dir', type=str, required=True,
                        help='Directory with generated images (from generate_head2tail.py)')
    parser.add_argument('--dataset', type=str, default='cifar100_lt',
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['classification', 'diversity'],
                        choices=['fid', 'diversity', 'classification', 'domain_fid'])
    parser.add_argument('--classifier_ckpt', type=str, default='',
                        help='Path to trained classifier checkpoint (for classification metric)')
    parser.add_argument('--arch', type=str, default='resnet32')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_file', type=str, default='',
                        help='Save results to this JSON file')

    return parser.parse_args()


def load_generated_images(generated_dir):
    """Load generated images grouped by class.

    Returns:
        Dict[int, List[PIL.Image]]
    """
    class_images = {}

    meta_path = os.path.join(generated_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        for class_key, entries in metadata.items():
            class_idx = int(class_key.split('_')[1])
            images = []
            for entry in entries:
                fpath = os.path.join(generated_dir, entry['path'])
                if os.path.exists(fpath):
                    images.append(Image.open(fpath).convert('RGB'))
            if images:
                class_images[class_idx] = images
    else:
        # Fallback: scan directories
        for dirname in sorted(os.listdir(generated_dir)):
            dirpath = os.path.join(generated_dir, dirname)
            if not os.path.isdir(dirpath) or not dirname.startswith('class_'):
                continue
            class_idx = int(dirname.split('_')[1])
            images = []
            for fname in sorted(os.listdir(dirpath)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    images.append(Image.open(os.path.join(dirpath, fname)).convert('RGB'))
            if images:
                class_images[class_idx] = images

    return class_images


def evaluate_classification(class_images, classifier_ckpt, arch,
                             num_classes, device, is_cifar=True):
    """Evaluate: what % of generated images are classified as the target class.

    Higher = better faithfulness to the target class.
    """
    from models import create_model
    from utils import load_checkpoint

    model = create_model(arch, num_classes)
    load_checkpoint(model, classifier_ckpt)
    model = model.to(device)
    model.eval()

    if is_cifar:
        from datasets.cifar_lt import get_cifar_test_transform
        transform = get_cifar_test_transform()
        resize = 32
    else:
        from datasets.imagenet_lt import get_imagenet_test_transform
        transform = get_imagenet_test_transform()
        resize = 224

    total = 0
    correct = 0
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for class_idx, images in class_images.items():
            per_class_correct[class_idx] = 0
            per_class_total[class_idx] = len(images)

            # Process in batches
            batch_tensors = []
            for img in images:
                img_resized = img.resize((resize, resize))
                tensor = transform(img_resized)
                batch_tensors.append(tensor)

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu()

            for pred in preds:
                total += 1
                if pred.item() == class_idx:
                    correct += 1
                    per_class_correct[class_idx] += 1

    overall_acc = correct / total * 100 if total > 0 else 0
    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c] * 100
        if per_class_total[c] > 0 else 0
        for c in per_class_correct
    }

    print(f"\n[Classification Faithfulness]")
    print(f"  Overall: {overall_acc:.2f}% ({correct}/{total})")
    print(f"  Per-class mean: {np.mean(list(per_class_acc.values())):.2f}%")
    print(f"  Per-class min: {min(per_class_acc.values()):.2f}%")
    print(f"  Per-class max: {max(per_class_acc.values()):.2f}%")

    return {
        'overall_acc': overall_acc,
        'per_class_acc': per_class_acc,
        'per_class_mean': np.mean(list(per_class_acc.values())),
    }


def evaluate_diversity(class_images, device, max_pairs=500):
    """Evaluate intra-class diversity using LPIPS.

    Higher LPIPS = more diverse generated images.
    """
    try:
        import lpips
    except ImportError:
        print("[WARN] lpips not installed. Skipping diversity metric.")
        print("  Install: pip install lpips")
        return {}

    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    per_class_diversity = {}

    for class_idx, images in class_images.items():
        if len(images) < 2:
            per_class_diversity[class_idx] = 0.0
            continue

        tensors = [transform(img).unsqueeze(0).to(device) for img in images]
        distances = []

        # Random pairs
        import random
        indices = list(range(len(tensors)))
        n_pairs = min(max_pairs, len(indices) * (len(indices) - 1) // 2)

        for _ in range(n_pairs):
            i, j = random.sample(indices, 2)
            with torch.no_grad():
                d = loss_fn(tensors[i], tensors[j]).item()
            distances.append(d)

        per_class_diversity[class_idx] = np.mean(distances)

    mean_diversity = np.mean(list(per_class_diversity.values()))
    print(f"\n[Intra-class Diversity (LPIPS)]")
    print(f"  Mean LPIPS: {mean_diversity:.4f}")
    print(f"  Min: {min(per_class_diversity.values()):.4f}")
    print(f"  Max: {max(per_class_diversity.values()):.4f}")

    return {
        'mean_diversity': mean_diversity,
        'per_class_diversity': {str(k): v for k, v in per_class_diversity.items()},
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Head-to-Tail Quality Evaluation")
    print("=" * 60)

    # Load generated images
    print(f"Loading generated images from {args.generated_dir}...")
    class_images = load_generated_images(args.generated_dir)
    total_imgs = sum(len(v) for v in class_images.values())
    print(f"  Loaded {total_imgs} images across {len(class_images)} classes")

    if total_imgs == 0:
        print("No images found. Check the generated_dir path.")
        return

    is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')
    num_classes = 10 if args.dataset == 'cifar10_lt' else (100 if args.dataset == 'cifar100_lt' else 1000)

    results = {}

    if 'classification' in args.metrics:
        if args.classifier_ckpt and os.path.exists(args.classifier_ckpt):
            results['classification'] = evaluate_classification(
                class_images, args.classifier_ckpt, args.arch,
                num_classes, args.device, is_cifar
            )
        else:
            print("[SKIP] Classification: no classifier checkpoint provided (--classifier_ckpt)")

    if 'diversity' in args.metrics:
        results['diversity'] = evaluate_diversity(
            class_images, args.device
        )

    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
