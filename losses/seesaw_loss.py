"""Seesaw Loss.

Reference: Wang et al., "Seesaw Loss for Long-Tailed Instance Segmentation",
CVPR 2021.

Two complementary factors:
- Mitigation factor: reduces penalty for being classified as a tail class (false positive)
- Compensation factor: increases gradient for hard tail-class samples
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeesawLoss(nn.Module):
    """Seesaw Loss for long-tailed recognition.

    Args:
        cls_num_list: List of per-class sample counts.
        p: Mitigation factor exponent (default: 0.8).
        q: Compensation factor exponent (default: 2.0).
    """

    def __init__(self, cls_num_list, p=0.8, q=2.0):
        super().__init__()
        self.p = p
        self.q = q
        cls_num = np.array(cls_num_list, dtype=np.float64)
        self.register_buffer('cls_num', torch.FloatTensor(cls_num))
        self.num_classes = len(cls_num_list)

    def forward(self, logits, targets):
        # Mitigation factor: S_ij = (n_j / n_i)^p for i != j
        # When n_i > n_j (i is head, j is tail), S_ij < 1 -> reduce penalty
        cls_num = self.cls_num.unsqueeze(0)  # (1, C)
        target_num = self.cls_num[targets].unsqueeze(1)  # (B, 1)
        mitigation = (cls_num / (target_num + 1e-12)).clamp(max=1.0) ** self.p

        # Compensation factor: based on predicted probability of target class
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1))  # (B, 1)
        compensation = (1 - target_probs) ** self.q

        # Combined seesaw weight
        seesaw_weights = mitigation * compensation

        # For the target class, weight should be 1.0
        one_hot = F.one_hot(targets, self.num_classes).float()
        seesaw_weights = seesaw_weights * (1 - one_hot) + one_hot

        # Weighted softmax
        adjusted_logits = logits + torch.log(seesaw_weights + 1e-12)
        return F.cross_entropy(adjusted_logits, targets)
