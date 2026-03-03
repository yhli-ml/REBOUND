"""LDAM Loss: Label-Distribution-Aware Margin Loss.

Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss", NeurIPS 2019.

Enforces larger margins for minority classes:
    margin_j = C / n_j^(1/4)
where n_j is the number of training samples in class j.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """LDAM Loss with optional Deferred Re-Weighting (DRW).

    Args:
        cls_num_list: List of per-class sample counts.
        max_m: Maximum margin (default: 0.5).
        s: Scaling factor (default: 30).
        weight: Optional per-class weight tensor (for DRW).
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        cls_num = np.array(cls_num_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.s = s
        self.weight = weight

    def forward(self, logits, targets):
        # Get per-sample margins
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)

        # Subtract margin from the correct class logit
        margin = self.m_list[targets].unsqueeze(1)
        logits_m = logits.clone()
        logits_m[index] -= margin.squeeze(1)

        # Scale and compute cross-entropy
        logits_m = logits_m * self.s

        return F.cross_entropy(logits_m, targets, weight=self.weight)

    def update_weight(self, weight):
        """Update class weights (for DRW - Deferred Re-Weighting)."""
        self.weight = weight


def get_drw_weights(cls_num_list, beta=0.9999):
    """Compute DRW weights based on effective number.

    Args:
        cls_num_list: List of per-class sample counts.
        beta: Smoothing parameter (default: 0.9999).

    Returns:
        torch.Tensor of per-class weights.
    """
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    return torch.FloatTensor(per_cls_weights)
