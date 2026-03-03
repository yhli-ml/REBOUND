"""Feature-space guided head class selection for Head-to-Tail transfer.

Core idea:
  For each tail class, find the K nearest head classes in feature space.
  The intuition is that head classes closer in feature space require smaller
  semantic transformations, leading to:
    1. More natural generated images (lower failure rate)
    2. Better domain consistency (less modification needed)
    3. Generated samples that fill feature space near the tail class boundary

Feature extraction options:
  - CLIP: Zero-shot, no training needed, good semantic understanding
  - Trained classifier: Dataset-specific features, captures learned boundaries
  - Both: Ensemble similarity for robust matching

Usage:
    selector = HeadClassSelector(dataset_name='cifar100_lt', device='cuda')
    selector.compute_prototypes(train_dataset)
    mapping = selector.get_head2tail_mapping(cls_num_list, top_k=3)
    # mapping[tail_class_idx] = [(head_class_idx, similarity_score), ...]
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict


class HeadClassSelector:
    """Select the best head classes as generation sources for each tail class.

    Supports two modes:
      1. CLIP-based: Uses CLIP text/image embeddings (no training required)
      2. Classifier-based: Uses features from a trained classifier

    Args:
        dataset_name: Dataset identifier for class name lookup.
        device: Torch device.
        feature_source: 'clip', 'classifier', or 'both'.
    """

    def __init__(self, dataset_name='cifar100_lt', device='cuda',
                 feature_source='clip'):
        self.dataset_name = dataset_name
        self.device = device
        self.feature_source = feature_source

        self.class_prototypes = None  # {class_idx: feature_vector}
        self.sample_features = None   # {class_idx: [feature_vectors]}

    # ------------------------------------------------------------------
    # CLIP-based prototype computation (zero-shot, no training needed)
    # ------------------------------------------------------------------
    def compute_clip_prototypes(self):
        """Compute class prototypes using CLIP text embeddings.

        Uses "a photo of a {class_name}" as text prompts and encodes them.
        This gives semantic similarity without needing any images.

        Returns:
            prototypes: Tensor of shape (num_classes, feat_dim)
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "Please install CLIP: pip install git+https://github.com/openai/CLIP.git"
            )

        from augment.head2tail_prompts import get_class_names

        class_names = get_class_names(self.dataset_name)
        num_classes = len(class_names)

        model, _ = clip.load('ViT-B/32', device=self.device)
        model.eval()

        # Encode class names as text
        text_prompts = [f"a photo of a {name.replace('_', ' ')}"
                        for name in class_names]
        text_tokens = clip.tokenize(text_prompts).to(self.device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        self.class_prototypes = {}
        for i in range(num_classes):
            self.class_prototypes[i] = text_features[i].cpu()

        self._prototype_tensor = text_features.cpu()  # (C, D)
        print(f"[HeadClassSelector] Computed CLIP text prototypes "
              f"for {num_classes} classes (dim={text_features.shape[1]})")

        return self._prototype_tensor

    def compute_clip_image_prototypes(self, train_dataset, batch_size=256):
        """Compute class prototypes using CLIP image embeddings.

        Encodes all training images through CLIP and averages per class.
        More accurate than text-only but requires forward pass on all images.

        Args:
            train_dataset: Dataset with .data (numpy) and .targets attributes.
            batch_size: Batch size for CLIP encoding.

        Returns:
            prototypes: Tensor of shape (num_classes, feat_dim)
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "Please install CLIP: pip install git+https://github.com/openai/CLIP.git"
            )
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset

        model, preprocess = clip.load('ViT-B/32', device=self.device)
        model.eval()

        # Wrap dataset for CLIP preprocessing
        class CLIPDataset(Dataset):
            def __init__(self, data, targets, preprocess):
                self.data = data
                self.targets = targets
                self.preprocess = preprocess

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img = self.data[idx]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                img = self.preprocess(img)
                return img, self.targets[idx]

        # Handle both CIFAR (numpy) and ImageNet (paths) datasets
        if hasattr(train_dataset, 'data'):
            data = train_dataset.data
        elif hasattr(train_dataset, 'img_paths'):
            data = train_dataset.img_paths
        else:
            data = [train_dataset[i][0] for i in range(len(train_dataset))]

        targets = list(train_dataset.targets)
        clip_ds = CLIPDataset(data, targets, preprocess)
        loader = DataLoader(clip_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        # Accumulate features per class
        num_classes = len(train_dataset.cls_num_list)
        feat_sums = {}
        feat_counts = {}
        self.sample_features = defaultdict(list)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                features = model.encode_image(images)
                features = F.normalize(features, dim=-1).cpu()

                for feat, label in zip(features, labels):
                    label = label.item() if isinstance(label, torch.Tensor) else label
                    if label not in feat_sums:
                        feat_sums[label] = torch.zeros_like(feat)
                        feat_counts[label] = 0
                    feat_sums[label] += feat
                    feat_counts[label] += 1
                    self.sample_features[label].append(feat)

        # Compute prototypes (mean features)
        self.class_prototypes = {}
        proto_list = []
        for c in range(num_classes):
            if c in feat_sums and feat_counts[c] > 0:
                proto = feat_sums[c] / feat_counts[c]
                proto = F.normalize(proto, dim=0)
            else:
                proto = torch.zeros(features.shape[-1])
            self.class_prototypes[c] = proto
            proto_list.append(proto)

        self._prototype_tensor = torch.stack(proto_list)  # (C, D)
        print(f"[HeadClassSelector] Computed CLIP image prototypes "
              f"for {num_classes} classes (dim={features.shape[-1]})")

        return self._prototype_tensor

    # ------------------------------------------------------------------
    # Classifier-based prototype computation
    # ------------------------------------------------------------------
    def compute_classifier_prototypes(self, model, train_dataset,
                                       batch_size=128, num_workers=4):
        """Compute class prototypes using a trained classifier's penultimate layer.

        Args:
            model: Trained classifier (nn.Module). Must have a `forward_features`
                   method or we hook into the layer before FC.
            train_dataset: Training dataset with transform applied.
            batch_size: Batch size.
            num_workers: DataLoader workers.

        Returns:
            prototypes: Tensor of shape (num_classes, feat_dim)
        """
        from torch.utils.data import DataLoader

        model = model.to(self.device)
        model.eval()

        # Register hook to capture features before FC
        features_store = []
        labels_store = []

        def hook_fn(module, input, output):
            # input[0] is the features feeding into FC
            features_store.append(input[0].detach().cpu())

        # Register hook on the FC layer
        fc = model.fc if hasattr(model, 'fc') else model.module.fc
        handle = fc.register_forward_pre_hook(hook_fn)

        loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    images, targets, _ = batch
                else:
                    images, targets = batch
                images = images.to(self.device)
                model(images)
                labels_store.append(targets)

        handle.remove()

        all_features = torch.cat(features_store, dim=0)  # (N, D)
        all_labels = torch.cat(labels_store, dim=0)       # (N,)

        # Average per class
        num_classes = len(train_dataset.cls_num_list)
        self.class_prototypes = {}
        self.sample_features = defaultdict(list)
        proto_list = []

        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                class_feats = all_features[mask]
                proto = F.normalize(class_feats.mean(dim=0), dim=0)
                self.sample_features[c] = [
                    F.normalize(f, dim=0) for f in class_feats
                ]
            else:
                proto = torch.zeros(all_features.shape[1])
            self.class_prototypes[c] = proto
            proto_list.append(proto)

        self._prototype_tensor = torch.stack(proto_list)
        print(f"[HeadClassSelector] Computed classifier prototypes "
              f"for {num_classes} classes (dim={all_features.shape[1]})")

        return self._prototype_tensor

    # ------------------------------------------------------------------
    # Head-to-Tail Mapping
    # ------------------------------------------------------------------
    def get_head2tail_mapping(self, cls_num_list, top_k=3,
                               head_threshold=100, tail_threshold=20):
        """For each tail class, find the K nearest head classes.

        Args:
            cls_num_list: List of per-class sample counts.
            top_k: Number of nearest head classes to select.
            head_threshold: Classes with > this many samples are "head".
            tail_threshold: Classes with <= this many samples are "tail".

        Returns:
            mapping: Dict[int, List[Tuple[int, float]]]
                {tail_class: [(head_class, similarity), ...]}
            head_classes: List[int] of head class indices.
            tail_classes: List[int] of tail class indices.
        """
        if self._prototype_tensor is None:
            raise RuntimeError("Call compute_*_prototypes() first!")

        num_classes = len(cls_num_list)
        prototypes = self._prototype_tensor.float()  # (C, D)

        # Identify head and tail classes
        head_classes = [c for c in range(num_classes)
                        if cls_num_list[c] > head_threshold]
        tail_classes = [c for c in range(num_classes)
                        if cls_num_list[c] <= tail_threshold]
        medium_classes = [c for c in range(num_classes)
                          if tail_threshold < cls_num_list[c] <= head_threshold]

        print(f"[HeadClassSelector] Head: {len(head_classes)}, "
              f"Medium: {len(medium_classes)}, Tail: {len(tail_classes)}")

        if not head_classes:
            print("[WARN] No head classes found! Using all non-tail classes as source.")
            head_classes = [c for c in range(num_classes)
                           if cls_num_list[c] > tail_threshold]

        if not tail_classes:
            print("[WARN] No tail classes found! Using medium classes as targets.")
            tail_classes = medium_classes

        # Compute pairwise similarities
        head_protos = prototypes[head_classes]  # (H, D)
        tail_protos = prototypes[tail_classes]  # (T, D)

        # Cosine similarity: (T, H)
        sim_matrix = torch.mm(
            F.normalize(tail_protos, dim=1),
            F.normalize(head_protos, dim=1).t()
        )

        mapping = {}
        for t_idx, tail_c in enumerate(tail_classes):
            sims = sim_matrix[t_idx]  # (H,)
            top_vals, top_idxs = torch.topk(sims, min(top_k, len(head_classes)))
            mapping[tail_c] = [
                (head_classes[hi.item()], sv.item())
                for hi, sv in zip(top_idxs, top_vals)
            ]

        return mapping, head_classes, tail_classes

    def get_nearest_head_samples(self, tail_class, head_class, n_samples=10):
        """Get specific head class samples nearest to the tail class prototype.

        Args:
            tail_class: Target tail class index.
            head_class: Source head class index.
            n_samples: Number of samples to return.

        Returns:
            List of (sample_index_in_class, similarity_score) tuples.
        """
        if self.sample_features is None or tail_class not in self.class_prototypes:
            raise RuntimeError("Call compute_*_prototypes() with image features first!")

        tail_proto = self.class_prototypes[tail_class]  # (D,)
        head_feats = self.sample_features.get(head_class, [])

        if not head_feats:
            return []

        head_feats_tensor = torch.stack(head_feats)  # (N, D)
        sims = F.cosine_similarity(
            head_feats_tensor, tail_proto.unsqueeze(0), dim=1
        )

        n = min(n_samples, len(head_feats))
        top_vals, top_idxs = torch.topk(sims, n)

        return [(idx.item(), val.item()) for idx, val in zip(top_idxs, top_vals)]

    def save_mapping(self, mapping, head_classes, tail_classes,
                      cls_num_list, save_path):
        """Save the head-to-tail mapping to a JSON file for inspection.

        Args:
            mapping: Output from get_head2tail_mapping().
            head_classes: Head class indices.
            tail_classes: Tail class indices.
            cls_num_list: Per-class counts.
            save_path: Output JSON path.
        """
        import json
        from augment.head2tail_prompts import get_class_names

        class_names = get_class_names(self.dataset_name)

        output = {
            "head_classes": {
                str(c): {
                    "name": class_names[c] if c < len(class_names) else f"class_{c}",
                    "count": cls_num_list[c]
                }
                for c in head_classes
            },
            "tail_classes": {
                str(c): {
                    "name": class_names[c] if c < len(class_names) else f"class_{c}",
                    "count": cls_num_list[c]
                }
                for c in tail_classes
            },
            "mapping": {
                str(tail_c): [
                    {
                        "head_class": hc,
                        "head_name": class_names[hc] if hc < len(class_names) else f"class_{hc}",
                        "similarity": round(sim, 4)
                    }
                    for hc, sim in pairs
                ]
                for tail_c, pairs in mapping.items()
            }
        }

        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"[HeadClassSelector] Mapping saved to {save_path}")
