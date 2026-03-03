"""Focal Loss for class-imbalanced learning.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Focal loss down-weights easy (well-classified) examples and focuses on
hard (misclassified) examples:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional class-balanced weighting.

    Args:
        cls_num_list: List of per-class sample counts (for class-based alpha).
                      If None, uniform weighting.
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples.
        alpha: Per-class weight. If None and cls_num_list is given, uses inverse frequency.
    """

    def __init__(self, cls_num_list=None, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma

        if alpha is not None:
            self.register_buffer('alpha', torch.FloatTensor(alpha))
        elif cls_num_list is not None:
            # Effective number weighting
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.register_buffer('alpha', torch.FloatTensor(per_cls_weights))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()
