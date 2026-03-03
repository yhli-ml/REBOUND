"""Class-Aware and Class-Balanced Samplers for long-tail learning.

These samplers are used in two-stage methods (e.g., cRT) where the
second stage trains the classifier with balanced sampling.
"""

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassAwareSampler(Sampler):
    """Class-Aware Sampler: samples each class with equal probability.

    At each step, uniformly pick a class, then uniformly pick a sample
    from that class. This results in balanced class representation.

    Args:
        dataset: Dataset with `targets` attribute.
        num_samples_per_cls: Number of samples per class per epoch.
                             If None, uses max(class_counts).
    """

    def __init__(self, dataset, num_samples_per_cls=None):
        self.targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
        self.num_classes = max(self.targets) + 1

        # Build per-class index lists
        self.class_indices = [[] for _ in range(self.num_classes)]
        for idx, target in enumerate(self.targets):
            self.class_indices[target].append(idx)

        if num_samples_per_cls is None:
            # Total samples = num_classes * max_samples_per_class
            max_count = max(len(indices) for indices in self.class_indices)
            self.num_samples_per_cls = max_count
        else:
            self.num_samples_per_cls = num_samples_per_cls

        self.num_samples = self.num_classes * self.num_samples_per_cls

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples_per_cls):
            # Random class order each time
            perm = np.random.permutation(self.num_classes)
            for cls_idx in perm:
                if len(self.class_indices[cls_idx]) == 0:
                    continue
                idx = np.random.choice(self.class_indices[cls_idx])
                indices.append(idx)
        # Shuffle to mix classes within each batch
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


class ClassBalancedSampler(Sampler):
    """Class-Balanced Sampler using inverse-frequency weights.

    Each sample is drawn with probability proportional to 1/n_c,
    where n_c is the number of samples in its class.

    Args:
        dataset: Dataset with `targets` attribute.
        num_samples: Total number of samples per epoch.
    """

    def __init__(self, dataset, num_samples=None):
        targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
        self.targets = np.array(targets)

        # Compute per-sample weights (inverse of class frequency)
        cls_counts = np.bincount(self.targets)
        per_cls_weight = 1.0 / (cls_counts + 1e-12)
        sample_weights = per_cls_weight[self.targets]
        sample_weights = sample_weights / sample_weights.sum()

        self.weights = torch.from_numpy(sample_weights).double()
        self.num_samples = num_samples if num_samples else len(targets)

    def __iter__(self):
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
