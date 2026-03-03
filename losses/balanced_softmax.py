"""Balanced Softmax Loss.

Reference: Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition",
NeurIPS 2020.

Adds log(n_y) to the logit of class y before softmax, providing an unbiased
estimate that accounts for the label distribution shift between training and test.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Cross-Entropy Loss.

    Args:
        cls_num_list: List of per-class sample counts.
    """

    def __init__(self, cls_num_list):
        super().__init__()
        cls_num = torch.FloatTensor(cls_num_list)
        self.register_buffer('log_cls_num', torch.log(cls_num + 1e-12))

    def forward(self, logits, targets):
        # Add log(class count) to logits
        adjusted_logits = logits + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)
