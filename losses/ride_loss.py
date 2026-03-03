"""RIDE Loss: Routing Diverse Distribution-Aware Experts.

Reference: Wang et al., "Long-tailed Recognition via Routing Diverse
Distribution-Aware Experts", ICLR 2021.

Combines LDAM-style margins with diversity regularization among experts.
This is a simplified version for single-expert baselines.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RIDELoss(nn.Module):
    """Simplified RIDE Loss (LDAM + effective number reweighting + optional diversity).

    Args:
        cls_num_list: List of per-class sample counts.
        max_m: Maximum margin (default: 0.5).
        s: Scaling factor (default: 30).
        reweight_epoch: Epoch to start applying DRW weights (default: -1, disabled).
        beta: Beta for effective number computation.
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30.0,
                 reweight_epoch=-1, beta=0.9999):
        super().__init__()
        cls_num = np.array(cls_num_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.s = s

        # DRW weights
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.register_buffer('per_cls_weights', torch.FloatTensor(per_cls_weights))

        self.reweight_epoch = reweight_epoch
        self.weight = None
        self.current_epoch = 0

    def update_epoch(self, epoch):
        """Called at the start of each epoch to update DRW weights."""
        self.current_epoch = epoch
        if self.reweight_epoch >= 0 and epoch >= self.reweight_epoch:
            self.weight = self.per_cls_weights
        else:
            self.weight = None

    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)

        margin = self.m_list[targets].unsqueeze(1)
        logits_m = logits.clone()
        logits_m[index] -= margin.squeeze(1)
        logits_m = logits_m * self.s

        return F.cross_entropy(logits_m, targets, weight=self.weight)
