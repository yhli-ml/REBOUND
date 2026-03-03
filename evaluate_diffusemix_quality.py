#!/usr/bin/env python3
"""
Diagnostic tool for evaluating DiffuseMix augmented image quality.

Quantifies domain shift between original and augmented images using:
  1. Pixel-level statistics (mean, std, histogram divergence)
  2. Feature-level FID (Fréchet Inception Distance) — gold standard
  3. Per-class FID breakdown (which classes are most shifted)
  4. Classifier-based domain discrimination (can a model tell orig vs aug?)
  5. Visual grid for qualitative inspection

Usage:
  python evaluate_diffusemix_quality.py \
      --dataset cifar100_lt --data_root /hss/giil/temp/data \
      --diffusemix_dir ./data/diffusemix_cifar100_lt_IF100 \
      --output_dir ./eval_diffusemix \
      --gpu 0

  # Quick pixel-level only (no GPU model needed):
  python evaluate_diffusemix_quality.py \
      --dataset cifar100_lt --data_root /hss/giil/temp/data \
      --diffusemix_dir ./data/diffusemix_cifar100_lt_IF100 \
      --output_dir ./eval_diffusemix \
      --skip_fid --skip_discriminator
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ========================
#  Dataset Loading
# ========================

def load_original_cifar(dataset_name, data_root, imb_factor=0.01):
    """Load original CIFAR-LT dataset, return per-class image lists."""
    sys.path.insert(0, os.path.dirname(__file__))
    from datasets.cifar_lt import IMBALANCECIFAR10, IMBALANCECIFAR100

    if dataset_name == 'cifar10_lt':
        ds = IMBALANCECIFAR10(root=data_root, train=True,
                              imb_factor=imb_factor, download=True)
    else:
        ds = IMBALANCECIFAR100(root=data_root, train=True,
                               imb_factor=imb_factor, download=True)

    class_images = defaultdict(list)
    for i in range(len(ds.data)):
        # ds.data[i] is numpy (H, W, C) uint8
        class_images[ds.targets[i]].append(ds.data[i])

    return class_images, ds.cls_num_list


def load_augmented_images(diffusemix_dir):
    """Load augmented images from DiffuseMix output directory."""
    class_images = defaultdict(list)

    meta_path = os.path.join(diffusemix_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        for class_key, entries in metadata.items():
            class_idx = int(class_key.split('_')[1])
            for entry in entries:
                full_path = os.path.join(diffusemix_dir, entry['path'])
                if os.path.exists(full_path):
                    img = np.array(Image.open(full_path).convert('RGB'))
                    class_images[class_idx].append(img)
    else:
        # Fallback: scan directories
        for dirname in sorted(os.listdir(diffusemix_dir)):
            dirpath = os.path.join(diffusemix_dir, dirname)
            if not os.path.isdir(dirpath) or not dirname.startswith('class_'):
                continue
            class_idx = int(dirname.split('_')[1])
            for fname in sorted(os.listdir(dirpath)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = np.array(Image.open(os.path.join(dirpath, fname)).convert('RGB'))
                    class_images[class_idx].append(img)

    return class_images


# ========================
#  1. Pixel-Level Analysis
# ========================

def pixel_level_analysis(orig_images, aug_images, output_dir):
    """Compare pixel-level statistics between original and augmented images."""
    print("\n" + "="*60)
    print(" 1. Pixel-Level Statistics")
    print("="*60)

    all_classes = sorted(set(orig_images.keys()) & set(aug_images.keys()))

    results = {}
    for cls in all_classes:
        orig = np.stack(orig_images[cls]).astype(np.float32) / 255.0
        aug = np.stack(aug_images[cls]).astype(np.float32) / 255.0

        results[cls] = {
            'orig_mean': orig.mean(axis=(0, 1, 2)).tolist(),
            'aug_mean': aug.mean(axis=(0, 1, 2)).tolist(),
            'orig_std': orig.std(axis=(0, 1, 2)).tolist(),
            'aug_std': aug.std(axis=(0, 1, 2)).tolist(),
            'mean_diff': float(np.abs(orig.mean() - aug.mean())),
            'std_diff': float(np.abs(orig.std() - aug.std())),
            'n_orig': len(orig_images[cls]),
            'n_aug': len(aug_images[cls]),
        }

    # Global statistics
    all_orig = np.concatenate([np.stack(orig_images[c]) for c in all_classes]).astype(np.float32) / 255.0
    all_aug = np.concatenate([np.stack(aug_images[c]) for c in all_classes]).astype(np.float32) / 255.0

    print(f"\n  Original images: n={len(all_orig)}")
    print(f"    Mean (RGB): {all_orig.mean(axis=(0,1,2))}")
    print(f"    Std  (RGB): {all_orig.std(axis=(0,1,2))}")
    print(f"\n  Augmented images: n={len(all_aug)}")
    print(f"    Mean (RGB): {all_aug.mean(axis=(0,1,2))}")
    print(f"    Std  (RGB): {all_aug.std(axis=(0,1,2))}")

    # Histogram comparison (KL divergence per channel)
    print("\n  Histogram KL Divergence (per channel):")
    kl_divs = []
    for ch, ch_name in enumerate(['R', 'G', 'B']):
        orig_hist, _ = np.histogram(all_orig[:, :, :, ch].flatten(), bins=64, range=(0, 1), density=True)
        aug_hist, _ = np.histogram(all_aug[:, :, :, ch].flatten(), bins=64, range=(0, 1), density=True)
        # Add epsilon for numerical stability
        orig_hist = orig_hist + 1e-10
        aug_hist = aug_hist + 1e-10
        orig_hist /= orig_hist.sum()
        aug_hist /= aug_hist.sum()
        kl = float(np.sum(orig_hist * np.log(orig_hist / aug_hist)))
        kl_divs.append(kl)
        print(f"    {ch_name}: KL = {kl:.4f}")
    print(f"    Avg KL: {np.mean(kl_divs):.4f}")
    print(f"    (KL < 0.01 = very similar, > 0.1 = significant shift)")

    # Per-class mean shift
    mean_diffs = [results[c]['mean_diff'] for c in all_classes]
    print(f"\n  Per-class mean brightness shift:")
    print(f"    Min: {min(mean_diffs):.4f}, Max: {max(mean_diffs):.4f}, "
          f"Avg: {np.mean(mean_diffs):.4f}")

    # Plot histogram comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ch, (ax, ch_name) in enumerate(zip(axes, ['Red', 'Green', 'Blue'])):
        ax.hist(all_orig[:, :, :, ch].flatten(), bins=64, range=(0, 1),
                alpha=0.5, label='Original', density=True, color='blue')
        ax.hist(all_aug[:, :, :, ch].flatten(), bins=64, range=(0, 1),
                alpha=0.5, label='Augmented', density=True, color='red')
        ax.set_title(f'{ch_name} Channel')
        ax.legend()
    plt.suptitle('Pixel Intensity Distribution: Original vs Augmented')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_histogram.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: pixel_histogram.png")

    return results, np.mean(kl_divs)


# ========================
#  2. FID (Feature-Level)
# ========================

class InceptionFeatureExtractor:
    """Extract features using InceptionV3 or a ResNet for FID computation."""

    def __init__(self, device='cuda', model_type='inception'):
        self.device = device
        if model_type == 'inception':
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            # Remove the final FC layer — we want 2048-dim features
            self.model.fc = nn.Identity()
            self.model.eval().to(device)
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            self.feat_dim = 2048
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.model.eval().to(device)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            self.feat_dim = 512

    @torch.no_grad()
    def extract_features(self, images_np, batch_size=64):
        """Extract features from numpy images (N, H, W, C) uint8."""
        all_feats = []
        for i in range(0, len(images_np), batch_size):
            batch = images_np[i:i+batch_size]
            tensors = []
            for img_np in batch:
                pil_img = Image.fromarray(img_np)
                tensors.append(self.transform(pil_img))
            batch_tensor = torch.stack(tensors).to(self.device)
            feats = self.model(batch_tensor)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)


def compute_fid(feats1, feats2):
    """Compute Fréchet Inception Distance between two feature sets."""
    from scipy import linalg

    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def fid_analysis(orig_images, aug_images, output_dir, device='cuda',
                 model_type='inception'):
    """Compute FID between original and augmented images, globally and per-class."""
    print("\n" + "="*60)
    print(f" 2. FID Analysis (using {model_type})")
    print("="*60)

    extractor = InceptionFeatureExtractor(device=device, model_type=model_type)
    all_classes = sorted(set(orig_images.keys()) & set(aug_images.keys()))

    # Global FID
    print("\n  Extracting features (original)...")
    all_orig = np.concatenate([np.stack(orig_images[c]) for c in all_classes])
    orig_feats = extractor.extract_features(all_orig)

    print("  Extracting features (augmented)...")
    all_aug = np.concatenate([np.stack(aug_images[c]) for c in all_classes])
    aug_feats = extractor.extract_features(all_aug)

    global_fid = compute_fid(orig_feats, aug_feats)
    print(f"\n  Global FID: {global_fid:.2f}")
    print(f"  (FID < 50 = good, 50-100 = moderate shift, > 100 = severe shift)")

    # Per-class FID (only for classes with enough samples)
    print("\n  Per-class FID (classes with >= 10 samples in both sets):")
    per_class_fid = {}
    orig_offset = 0
    aug_offset = 0

    # Re-extract per-class (use offsets from the global extraction)
    orig_class_bounds = {}
    aug_class_bounds = {}
    o_idx = 0
    a_idx = 0
    for c in all_classes:
        n_o = len(orig_images[c])
        n_a = len(aug_images[c])
        orig_class_bounds[c] = (o_idx, o_idx + n_o)
        aug_class_bounds[c] = (a_idx, a_idx + n_a)
        o_idx += n_o
        a_idx += n_a

    for c in all_classes:
        o_start, o_end = orig_class_bounds[c]
        a_start, a_end = aug_class_bounds[c]
        n_o = o_end - o_start
        n_a = a_end - a_start

        if n_o < 10 or n_a < 10:
            continue

        c_orig_feats = orig_feats[o_start:o_end]
        c_aug_feats = aug_feats[a_start:a_end]

        try:
            c_fid = compute_fid(c_orig_feats, c_aug_feats)
            per_class_fid[c] = c_fid
        except Exception:
            pass

    if per_class_fid:
        fid_values = list(per_class_fid.values())
        print(f"    Computed for {len(per_class_fid)} classes")
        print(f"    Min FID: {min(fid_values):.2f} (class {min(per_class_fid, key=per_class_fid.get)})")
        print(f"    Max FID: {max(fid_values):.2f} (class {max(per_class_fid, key=per_class_fid.get)})")
        print(f"    Mean FID: {np.mean(fid_values):.2f}")
        print(f"    Median FID: {np.median(fid_values):.2f}")

        # Top-10 worst classes
        sorted_fid = sorted(per_class_fid.items(), key=lambda x: x[1], reverse=True)
        print(f"\n    Top-10 most shifted classes:")
        for c, fid_val in sorted_fid[:10]:
            n_o = len(orig_images[c])
            n_a = len(aug_images[c])
            print(f"      Class {c:3d}: FID = {fid_val:7.2f} "
                  f"(orig: {n_o}, aug: {n_a})")

        # Plot per-class FID
        fig, ax = plt.subplots(figsize=(12, 5))
        classes_sorted = sorted(per_class_fid.keys())
        fids_sorted = [per_class_fid[c] for c in classes_sorted]
        orig_counts = [len(orig_images[c]) for c in classes_sorted]

        ax.bar(range(len(classes_sorted)), fids_sorted, alpha=0.7)
        ax.set_xlabel('Class Index')
        ax.set_ylabel('FID')
        ax.set_title('Per-Class FID (Original vs Augmented)')

        # Add a secondary axis for original class count
        ax2 = ax.twinx()
        ax2.plot(range(len(classes_sorted)), orig_counts, 'r-', alpha=0.5,
                 label='Original count')
        ax2.set_ylabel('Original Sample Count', color='r')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_fid.png'), dpi=150)
        plt.close()
        print(f"\n  Saved: per_class_fid.png")

    # Correlation: FID vs original class count
    if per_class_fid:
        fid_vs_count = [(len(orig_images[c]), per_class_fid[c])
                        for c in per_class_fid]
        counts, fids = zip(*fid_vs_count)
        corr = np.corrcoef(counts, fids)[0, 1]
        print(f"\n  Correlation(original_count, FID) = {corr:.3f}")
        print(f"  (Negative = tail classes are MORE shifted, which is bad)")

    return global_fid, per_class_fid


# ========================
#  3. Domain Discriminator
# ========================

class DomainDataset(Dataset):
    """Binary dataset: original (label=0) vs augmented (label=1)."""
    def __init__(self, orig_images_list, aug_images_list, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for img in orig_images_list:
            self.images.append(img)
            self.labels.append(0)
        for img in aug_images_list:
            self.images.append(img)
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def domain_discriminator_analysis(orig_images, aug_images, output_dir,
                                  device='cuda'):
    """Train a simple classifier to distinguish original vs augmented.

    If it achieves high accuracy, the domain shift is severe.
    If it's near 50%, the augmented images are indistinguishable.
    """
    print("\n" + "="*60)
    print(" 3. Domain Discriminator Test")
    print("="*60)
    print("  (Can a classifier tell original from augmented?)")

    all_classes = sorted(set(orig_images.keys()) & set(aug_images.keys()))

    # Collect all images
    all_orig = [img for c in all_classes for img in orig_images[c]]
    all_aug = [img for c in all_classes for img in aug_images[c]]

    # Balance the sets
    min_n = min(len(all_orig), len(all_aug))
    np.random.seed(42)
    idx_o = np.random.permutation(len(all_orig))[:min_n]
    idx_a = np.random.permutation(len(all_aug))[:min_n]
    sel_orig = [all_orig[i] for i in idx_o]
    sel_aug = [all_aug[i] for i in idx_a]

    # Split train/test (80/20)
    n_train = int(0.8 * min_n)
    train_orig = sel_orig[:n_train]
    train_aug = sel_aug[:n_train]
    test_orig = sel_orig[n_train:]
    test_aug = sel_aug[n_train:]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = DomainDataset(train_orig, train_aug, transform)
    test_ds = DomainDataset(test_orig, test_aug, transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # Simple CNN discriminator
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    print(f"\n  Training discriminator ({n_train}x2 train, {min_n-n_train}x2 test)...")
    for epoch in range(20):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_probs.extend(probs.cpu().numpy().tolist())

    disc_acc = correct / total * 100
    print(f"\n  Discriminator Accuracy: {disc_acc:.1f}%")
    print(f"  Interpretation:")
    if disc_acc < 55:
        print(f"    EXCELLENT — augmented images are nearly indistinguishable")
    elif disc_acc < 70:
        print(f"    GOOD — moderate distributional difference, acceptable")
    elif disc_acc < 85:
        print(f"    CONCERNING — noticeable domain shift")
    else:
        print(f"    SEVERE — augmented images are clearly from a different domain")
        print(f"    This likely explains the performance degradation.")

    return disc_acc


# ========================
#  4. Visual Grid
# ========================

def save_visual_grid(orig_images, aug_images, output_dir, n_classes=10,
                     n_per_class=5):
    """Save a visual comparison grid: original vs augmented for each class."""
    print("\n" + "="*60)
    print(" 4. Visual Comparison Grid")
    print("="*60)

    all_classes = sorted(set(orig_images.keys()) & set(aug_images.keys()))

    # Pick classes spread across the distribution (head, mid, tail)
    if len(all_classes) > n_classes:
        indices = np.linspace(0, len(all_classes) - 1, n_classes, dtype=int)
        selected = [all_classes[i] for i in indices]
    else:
        selected = all_classes

    fig, axes = plt.subplots(len(selected), n_per_class * 2 + 1,
                              figsize=(n_per_class * 4 + 1, len(selected) * 2.2))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for row, cls in enumerate(selected):
        origs = orig_images[cls]
        augs = aug_images[cls]

        # Label column
        axes[row, 0].text(0.5, 0.5,
                          f"Class {cls}\norig:{len(origs)}\naug:{len(augs)}",
                          ha='center', va='center', fontsize=8,
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')

        # Original samples
        for j in range(n_per_class):
            ax = axes[row, j + 1]
            if j < len(origs):
                ax.imshow(origs[j])
                if row == 0:
                    ax.set_title('Orig', fontsize=7)
            ax.axis('off')

        # Augmented samples
        for j in range(n_per_class):
            ax = axes[row, n_per_class + 1 + j]
            if j < len(augs):
                ax.imshow(augs[j])
                if row == 0:
                    ax.set_title('Aug', fontsize=7)
            ax.axis('off')

    plt.suptitle('Original (left) vs DiffuseMix Augmented (right)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visual_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: visual_comparison.png")


# ========================
#  5. Ablation Helper
# ========================

def print_ablation_suggestions(global_fid, disc_acc, kl_div):
    """Based on the diagnostics, suggest what to try next."""
    print("\n" + "="*60)
    print(" 5. Diagnosis & Recommendations")
    print("="*60)

    severe = False
    if disc_acc is not None and disc_acc > 80:
        severe = True
    if global_fid is not None and global_fid > 100:
        severe = True

    if severe:
        print("""
  DIAGNOSIS: Severe domain shift detected.
  The augmented images are very different from original CIFAR images.

  Root causes (likely):
    a) InstructPix2Pix at 256×256 → resize to 32×32 destroys CIFAR texture
    b) Style prompts (sunset/Autumn/watercolor) change color distribution
    c) Fractal blending adds noise unrelated to class semantics
    d) Concatenation (half orig + half styled) creates unnatural patterns

  Recommended ablation experiments (add to your script):

    1) ONLY-AUG: Train ONLY on augmented data → how bad is it alone?
       python main.py ... --only_aug  (need to implement this flag)

    2) AUG-RATIO: Mix original + 10%/25%/50% of augmented data
       → Find the dose where performance starts dropping

    3) AUG-WEIGHT: Use loss weighting for augmented samples
       → Give augmented images 0.3x weight in loss

    4) SOFTER-AUGMENT: Reduce fractal_alpha to 0.05, remove concat,
       use milder prompts like "natural lighting"

    5) TAIL-ONLY: Only augment classes with < 20 samples
       → Avoid contaminating classes that already have enough data

    6) FEATURE-ALIGN: After generating, filter out images whose
       features are too far from original class centroid
""")
    else:
        print("""
  DIAGNOSIS: Domain shift is moderate/acceptable.
  The degradation may be from other factors.

  Possible causes:
    a) The balanced distribution after augmentation changes the effective
       decision boundary — head classes lose their advantage
    b) Label noise in generated images (wrong class content)
    c) Hyperparameter mismatch (lr_schedule, weight_decay)

  Recommended next steps:
    1) Fix hyperparameters (lr_schedule=cosine, weight_decay=0.0002)
    2) Check per-class accuracy: are head classes dropping a lot?
    3) Try augmenting ONLY few-shot classes (< 20 samples)
""")


# ========================
#  Main
# ========================

def main():
    parser = argparse.ArgumentParser(description='Evaluate DiffuseMix Quality')
    parser.add_argument('--dataset', type=str, default='cifar100_lt')
    parser.add_argument('--data_root', type=str, default='/hss/giil/temp/data')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--diffusemix_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_diffusemix')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID computation (needs GPU + inception)')
    parser.add_argument('--skip_discriminator', action='store_true',
                        help='Skip domain discriminator test')
    parser.add_argument('--fid_model', type=str, default='inception',
                        choices=['inception', 'resnet18'],
                        help='Model for FID feature extraction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print(" DiffuseMix Quality Evaluation")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  DiffuseMix dir: {args.diffusemix_dir}")
    print(f"  Output: {args.output_dir}")

    # Load data
    print("\n  Loading original dataset...")
    orig_images, cls_num_list = load_original_cifar(
        args.dataset, args.data_root, args.imb_factor)
    print(f"  Original: {sum(len(v) for v in orig_images.values())} images, "
          f"{len(orig_images)} classes")

    print("  Loading augmented dataset...")
    aug_images = load_augmented_images(args.diffusemix_dir)
    print(f"  Augmented: {sum(len(v) for v in aug_images.values())} images, "
          f"{len(aug_images)} classes")

    # 1. Pixel analysis
    pixel_results, kl_div = pixel_level_analysis(orig_images, aug_images, args.output_dir)

    # 2. FID
    global_fid = None
    per_class_fid = None
    if not args.skip_fid:
        global_fid, per_class_fid = fid_analysis(
            orig_images, aug_images, args.output_dir,
            device=device, model_type=args.fid_model)
    else:
        print("\n  [Skipped FID analysis]")

    # 3. Domain discriminator
    disc_acc = None
    if not args.skip_discriminator:
        disc_acc = domain_discriminator_analysis(
            orig_images, aug_images, args.output_dir, device=device)
    else:
        print("\n  [Skipped domain discriminator]")

    # 4. Visual grid
    save_visual_grid(orig_images, aug_images, args.output_dir)

    # 5. Summary & recommendations
    print_ablation_suggestions(global_fid, disc_acc, kl_div)

    # Save summary
    summary = {
        'kl_divergence': float(kl_div),
        'global_fid': float(global_fid) if global_fid is not None else None,
        'discriminator_acc': float(disc_acc) if disc_acc is not None else None,
        'n_original': sum(len(v) for v in orig_images.values()),
        'n_augmented': sum(len(v) for v in aug_images.values()),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {args.output_dir}/summary.json")


if __name__ == '__main__':
    main()
