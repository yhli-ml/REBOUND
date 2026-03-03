"""Mixup, CutMix, and Remix data augmentation for long-tail learning.

References:
- Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
- CutMix: Yun et al., "CutMix: Regularization Strategy...", ICCV 2019.
- Remix: Chou et al., "Remix: Rebalanced Mixup", ECCV 2020 Workshop.
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation.

    Args:
        x: Input images (B, C, H, W).
        y: Labels (B,).
        alpha: Beta distribution parameter.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss.

    Args:
        criterion: Loss function.
        pred: Model predictions.
        y_a, y_b: Original and shuffled labels.
        lam: Mixup interpolation ratio.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation.

    Args:
        x: Input images (B, C, H, W).
        y: Labels (B,).
        alpha: Beta distribution parameter.

    Returns:
        mixed_x, y_a, y_b, lam (actual area ratio)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Actual lambda based on pixel area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def remix_data(x, y, cls_num_list, alpha=1.0, kappa=3.0, tau=0.5):
    """Remix augmentation (rebalanced mixup for long-tail learning).

    Adjusts the mixup ratio based on class frequency: if the minority class
    is mixed, give it a larger weight.

    Args:
        x: Input images (B, C, H, W).
        y: Labels (B,).
        cls_num_list: List of per-class sample counts.
        alpha: Beta distribution parameter.
        kappa: Threshold ratio for remix rebalancing.
        tau: Minimum lambda for the minority class.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    cls_num = torch.FloatTensor(cls_num_list).to(x.device)
    n_a = cls_num[y]
    n_b = cls_num[y[index]]

    # If n_a / n_b >= kappa (a is head, b is tail), give more weight to b
    # If n_b / n_a >= kappa (b is head, a is tail), give more weight to a
    lam_tensor = torch.full((batch_size,), lam, device=x.device)

    head_a = (n_a / (n_b + 1e-12)) >= kappa
    lam_tensor[head_a] = torch.clamp(lam_tensor[head_a], max=1.0 - tau)

    head_b = (n_b / (n_a + 1e-12)) >= kappa
    lam_tensor[head_b] = torch.clamp(lam_tensor[head_b], min=tau)

    lam_col = lam_tensor.view(-1, 1, 1, 1)
    mixed_x = lam_col * x + (1 - lam_col) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam_tensor
