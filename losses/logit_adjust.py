"""Logit Adjustment (LA) Loss.

Reference: Menon et al., "Long-tail Learning via Logit Adjustment", ICLR 2021.

Adjusts logits by adding tau * log(pi_y) to the logit of class y,
where pi_y is the class prior (estimated from training set frequencies).

Supports both:
- Training-time LA: modifies logits during training
- Post-hoc LA: adjusts logits at inference time (use `logit_adjust_posthoc()`)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedLoss(nn.Module):
    """Logit-Adjusted Cross-Entropy Loss.

    Args:
        cls_num_list: List of per-class sample counts.
        tau: Temperature for logit adjustment (default: 1.0).
    """

    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num = np.array(cls_num_list, dtype=np.float64)
        cls_prior = cls_num / cls_num.sum()
        log_prior = torch.FloatTensor(np.log(cls_prior + 1e-12))
        self.register_buffer('log_prior', log_prior)
        self.tau = tau

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)


def logit_adjust_posthoc(logits, cls_num_list, tau=1.0):
    """Post-hoc logit adjustment at inference time.

    Args:
        logits: Raw model logits (B, C).
        cls_num_list: List of per-class sample counts.
        tau: Temperature.

    Returns:
        Adjusted logits.
    """
    cls_num = np.array(cls_num_list, dtype=np.float64)
    cls_prior = cls_num / cls_num.sum()
    log_prior = torch.FloatTensor(np.log(cls_prior + 1e-12)).to(logits.device)
    return logits - tau * log_prior.unsqueeze(0)
