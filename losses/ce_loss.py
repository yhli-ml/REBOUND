"""Cross-Entropy and Class-Balanced Cross-Entropy losses."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss (ERM baseline)."""

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight)


class ClassBalancedCELoss(nn.Module):
    """Class-Balanced Loss based on Effective Number of Samples.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number
    of Samples", CVPR 2019.

    Reweights CE loss using effective number: E_n = (1 - beta^n) / (1 - beta)

    Args:
        cls_num_list: List of per-class sample counts.
        beta: Hyperparameter in [0, 1). Common values: 0.9, 0.99, 0.999, 0.9999.
        loss_type: 'ce' for cross-entropy, 'focal' for focal loss variant.
        gamma: Focal loss gamma (only used when loss_type='focal').
    """

    def __init__(self, cls_num_list, beta=0.9999, loss_type='ce', gamma=2.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.register_buffer(
            'per_cls_weights',
            torch.FloatTensor(per_cls_weights)
        )

    def forward(self, logits, targets):
        if self.loss_type == 'focal':
            # Focal variant
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            p = torch.exp(-ce_loss)
            focal_weight = (1 - p) ** self.gamma
            loss = focal_weight * ce_loss
            # Apply class weights
            weights = self.per_cls_weights[targets]
            loss = (loss * weights).mean()
        else:
            loss = F.cross_entropy(logits, targets, weight=self.per_cls_weights)
        return loss
