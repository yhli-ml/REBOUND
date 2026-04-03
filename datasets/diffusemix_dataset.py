"""DiffuseMix-augmented dataset wrapper for long-tail learning.

Wraps an existing long-tail dataset and adds pre-generated DiffuseMix images.
Used during training (Stage 2 in the pipeline: after offline generation).

The augmented images are loaded from a directory produced by
`generate_diffusemix.py`, which contains:
  - metadata.json: mapping from class dirs to file paths + labels
  - augmented_list.txt: simple "path label" format
  - class_XXXX/ directories with augmented .jpg files
"""

import os
import json

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DiffuseMixDataset(Dataset):
    """Combines original LT dataset with pre-generated DiffuseMix images.

    The augmented images are stored on disk under `diffusemix_dir`.
    During training, both original and augmented images are served.

    The cls_num_list is updated to reflect the combined distribution.

    Args:
        original_dataset: The base long-tail dataset (has .targets, .__len__,
                          .__getitem__, .cls_num_list attributes).
        diffusemix_dir: Directory with pre-generated augmented images.
        transform: Transform to apply (same as original dataset's training
                   transform). Applied to augmented images loaded from disk.
        is_cifar: If True, original dataset returns numpy arrays, and
                  augmented images will be resized to 32x32.
        aug_img_size: Size to crop/resize augmented images before transform.
                      For CIFAR: 32. For ImageNet: None (keep original size).
    """

    def __init__(self, original_dataset, diffusemix_dir, transform=None,
                 is_cifar=False, aug_img_size=None, sample_ratio=1.0):
        self.original_dataset = original_dataset
        self.diffusemix_dir = diffusemix_dir
        self.transform = transform
        self.is_cifar = is_cifar
        self.aug_img_size = aug_img_size

        # Load augmented image metadata
        self.aug_samples = []  # list of (path, label)
        self._load_augmented_data()

        # Subsample augmented data if ratio < 1.0
        if 0.0 < sample_ratio < 1.0:
            import random
            random.seed(42)
            n_keep = max(1, int(len(self.aug_samples) * sample_ratio))
            self.aug_samples = random.sample(self.aug_samples, n_keep)
            print(f"[DiffuseMixDataset] Subsampled augmented data: "
                  f"ratio={sample_ratio}, kept {n_keep}/{len(self.aug_samples)+n_keep}")

        # Build combined targets and cls_num_list
        self.n_original = len(original_dataset)
        self.n_augmented = len(self.aug_samples)
        self.total_len = self.n_original + self.n_augmented

        # Combine targets
        orig_targets = list(original_dataset.targets)
        aug_targets = [label for _, label in self.aug_samples]
        self.targets = orig_targets + aug_targets

        # Update cls_num_list
        orig_cls_num = list(original_dataset.cls_num_list)
        num_classes = len(orig_cls_num)
        aug_counts = np.zeros(num_classes, dtype=int)
        for label in aug_targets:
            if label < num_classes:
                aug_counts[label] += 1

        self.cls_num_list = [
            orig_cls_num[c] + aug_counts[c] for c in range(num_classes)
        ]

        print(f"[DiffuseMixDataset] Original: {self.n_original}, "
              f"Augmented: {self.n_augmented}, Total: {self.total_len}")
        print(f"[DiffuseMixDataset] Class counts range: "
              f"{min(self.cls_num_list)} ~ {max(self.cls_num_list)}")

    def _load_augmented_data(self):
        """Load augmented image paths from metadata."""
        # Try metadata.json first
        meta_path = os.path.join(self.diffusemix_dir, 'metadata.json')
        txt_path = os.path.join(self.diffusemix_dir, 'augmented_list.txt')

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            for class_key in metadata:
                for entry in metadata[class_key]:
                    full_path = os.path.join(self.diffusemix_dir, entry['path'])
                    if os.path.exists(full_path):
                        self.aug_samples.append((full_path, entry['label']))
        elif os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        rel_path, label = parts[0], int(parts[1])
                        full_path = os.path.join(self.diffusemix_dir, rel_path)
                        if os.path.exists(full_path):
                            self.aug_samples.append((full_path, label))
        else:
            # Fallback: scan directory structure class_XXXX/<images>
            for dirname in sorted(os.listdir(self.diffusemix_dir)):
                dirpath = os.path.join(self.diffusemix_dir, dirname)
                if not os.path.isdir(dirpath) or not dirname.startswith('class_'):
                    continue
                try:
                    class_idx = int(dirname.split('_')[1])
                except (ValueError, IndexError):
                    continue
                for fname in sorted(os.listdir(dirpath)):
                    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                        full_path = os.path.join(dirpath, fname)
                        self.aug_samples.append((full_path, class_idx))

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < self.n_original:
            # Original dataset sample
            img, label = self.original_dataset[index]
            return img, label, 0  # 0 = original
        else:
            # Augmented sample
            aug_idx = index - self.n_original
            img_path, label = self.aug_samples[aug_idx]
            img = Image.open(img_path).convert('RGB')

            # Resize for CIFAR
            if self.aug_img_size is not None:
                img = img.resize((self.aug_img_size, self.aug_img_size))

            # Apply same transform as training
            if self.transform is not None:
                img = self.transform(img)

            return img, label, 1  # 1 = augmented

    def get_cls_num_list(self):
        return self.cls_num_list
