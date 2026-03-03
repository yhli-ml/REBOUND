from .ce_loss import CrossEntropyLoss, ClassBalancedCELoss
from .focal_loss import FocalLoss
from .ldam_loss import LDAMLoss
from .logit_adjust import LogitAdjustedLoss
from .balanced_softmax import BalancedSoftmaxLoss
from .mixup import mixup_data, mixup_criterion, cutmix_data, remix_data
from .seesaw_loss import SeesawLoss
from .ride_loss import RIDELoss

__all__ = [
    'CrossEntropyLoss', 'ClassBalancedCELoss',
    'FocalLoss', 'LDAMLoss', 'LogitAdjustedLoss',
    'BalancedSoftmaxLoss', 'SeesawLoss', 'RIDELoss',
    'mixup_data', 'mixup_criterion', 'cutmix_data', 'remix_data',
    'get_loss',
]


def get_loss(name, cls_num_list, args=None):
    """Factory function to get a loss by name.

    Args:
        name: Loss name.
        cls_num_list: List of per-class sample counts.
        args: argparse namespace with additional parameters.

    Returns:
        Loss module.
    """
    name = name.lower()

    if name == 'ce':
        return CrossEntropyLoss()
    elif name == 'ce_weighted':
        return ClassBalancedCELoss(cls_num_list, beta=0.9999)
    elif name == 'cb_ce':
        beta = getattr(args, 'cb_beta', 0.9999) if args else 0.9999
        return ClassBalancedCELoss(cls_num_list, beta=beta)
    elif name == 'focal':
        gamma = getattr(args, 'focal_gamma', 2.0) if args else 2.0
        return FocalLoss(cls_num_list=cls_num_list, gamma=gamma)
    elif name == 'ldam':
        max_m = getattr(args, 'ldam_max_m', 0.5) if args else 0.5
        s = getattr(args, 'ldam_s', 30.0) if args else 30.0
        return LDAMLoss(cls_num_list, max_m=max_m, s=s)
    elif name == 'logit_adjust' or name == 'la':
        tau = getattr(args, 'la_tau', 1.0) if args else 1.0
        return LogitAdjustedLoss(cls_num_list, tau=tau)
    elif name == 'balanced_softmax' or name == 'bs':
        return BalancedSoftmaxLoss(cls_num_list)
    elif name == 'seesaw':
        p = getattr(args, 'seesaw_p', 0.8) if args else 0.8
        q = getattr(args, 'seesaw_q', 2.0) if args else 2.0
        return SeesawLoss(cls_num_list, p=p, q=q)
    else:
        raise ValueError(
            f"Unknown loss: {name}. Supported: "
            f"ce, ce_weighted, cb_ce, focal, ldam, logit_adjust/la, "
            f"balanced_softmax/bs, seesaw"
        )
