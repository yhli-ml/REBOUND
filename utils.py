"""Utility functions for long-tail learning baseline."""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn


# ========== Evaluation Metrics ==========

def get_class_split_lists(cls_num_list, dataset_name=None,
                          many_thr=100, few_thr=20,
                          head_ratio=0.3, tail_ratio=0.3):
    """Return head/medium/tail class ids.

    CIFAR-LT uses a rank-based split:
      - Head: top 30% classes by training count
      - Medium: middle 40%
      - Tail: bottom 30%

    Other datasets keep the standard count-threshold split.
    """
    num_classes = len(cls_num_list)

    if dataset_name in ('cifar10_lt', 'cifar100_lt'):
        sorted_classes = sorted(
            range(num_classes),
            key=lambda class_idx: (-cls_num_list[class_idx], class_idx)
        )
        n_head = int(num_classes * head_ratio)
        n_tail = int(num_classes * tail_ratio)
        n_medium = num_classes - n_head - n_tail

        head_classes = sorted_classes[:n_head]
        medium_classes = sorted_classes[n_head:n_head + n_medium]
        tail_classes = sorted_classes[n_head + n_medium:]
        return head_classes, medium_classes, tail_classes

    head_classes = [c for c in range(num_classes) if cls_num_list[c] > many_thr]
    medium_classes = [c for c in range(num_classes)
                      if few_thr < cls_num_list[c] <= many_thr]
    tail_classes = [c for c in range(num_classes) if cls_num_list[c] <= few_thr]
    return head_classes, medium_classes, tail_classes

def shot_acc(preds, labels, cls_num_list, many_shot_thr=100, low_shot_thr=20,
             dataset_name=None):
    """Compute accuracy for Many-shot, Medium-shot, and Few-shot classes.

    Following the standard protocol:
    - Many-shot: classes with > many_shot_thr training samples
    - Medium-shot: classes with low_shot_thr < samples <= many_shot_thr
    - Few-shot: classes with <= low_shot_thr training samples

    Args:
        preds: Predicted class indices (numpy array).
        labels: Ground-truth labels (numpy array).
        cls_num_list: List of per-class training sample counts.
        many_shot_thr: Threshold for many-shot classes (default: 100).
        low_shot_thr: Threshold for few-shot classes (default: 20).

    Returns:
        dict with keys: 'overall', 'many', 'medium', 'few'
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_classes = len(cls_num_list)

    # Per-class accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=many_shot_thr,
        few_thr=low_shot_thr,
    )
    head_set = set(head_classes)
    medium_set = set(medium_classes)
    tail_set = set(tail_classes)

    # Split into shots
    many_acc, medium_acc, few_acc = [], [], []
    for i in range(num_classes):
        if class_total[i] == 0:
            continue
        acc = class_correct[i] / class_total[i]
        if i in head_set:
            many_acc.append(acc)
        elif i in medium_set:
            medium_acc.append(acc)
        elif i in tail_set:
            few_acc.append(acc)

    overall = (preds == labels).sum() / len(labels) * 100

    result = {
        'overall': overall,
        'many': np.mean(many_acc) * 100 if many_acc else 0.0,
        'medium': np.mean(medium_acc) * 100 if medium_acc else 0.0,
        'few': np.mean(few_acc) * 100 if few_acc else 0.0,
    }
    return result


def per_class_accuracy(preds, labels, num_classes):
    """Compute per-class accuracy.

    Returns:
        numpy array of shape (num_classes,) with per-class accuracy.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    # Avoid division by zero
    mask = class_total > 0
    acc = np.zeros(num_classes)
    acc[mask] = class_correct[mask] / class_total[mask]
    return acc


# ========== Learning Rate Scheduling ==========

def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate based on schedule.

    Supports: 'cosine', 'step', 'warmup_step', 'warmup_cosine'
    """
    lr = args.lr

    if args.lr_schedule == 'cosine':
        lr = lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_schedule == 'step':
        # Decay at specified milestones
        milestones = getattr(args, 'lr_milestones', [160, 180])
        gamma = getattr(args, 'lr_gamma', 0.1)
        for milestone in milestones:
            if epoch >= milestone:
                lr *= gamma
    elif args.lr_schedule == 'warmup_step':
        warmup_epochs = getattr(args, 'warmup_epochs', 5)
        if epoch < warmup_epochs:
            lr = lr * (epoch + 1) / warmup_epochs
        else:
            milestones = getattr(args, 'lr_milestones', [160, 180])
            gamma = getattr(args, 'lr_gamma', 0.1)
            for milestone in milestones:
                if epoch >= milestone:
                    lr *= gamma
    elif args.lr_schedule == 'warmup_cosine':
        warmup_epochs = getattr(args, 'warmup_epochs', 5)
        if epoch < warmup_epochs:
            lr = lr * (epoch + 1) / warmup_epochs
        else:
            lr = lr * 0.5 * (1.0 + math.cos(
                math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            ))
    elif args.lr_schedule == 'linear':
        lr = lr * (1.0 - epoch / args.epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# ========== Checkpoint Utilities ==========

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Save checkpoint and optionally copy to best."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, strict=True):
    """Load checkpoint into model and optionally optimizer.

    Returns:
        dict with 'epoch' and 'best_acc' if available.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=strict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    info = {}
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'best_acc' in checkpoint:
        info['best_acc'] = checkpoint['best_acc']
    return info


# ========== Tau-Normalization (Post-hoc) ==========

def tau_normalize(model, tau=1.0):
    """Apply tau-normalization to the classifier weights.

    Reference: Kang et al., "Decoupling Representation and Classifier
    for Long-Tailed Recognition", ICLR 2020.

    Normalizes classifier weights: w_i = w_i / ||w_i||^tau

    Args:
        model: Model with `fc` attribute (the classifier).
        tau: Normalization temperature. tau=1.0 gives L2-normalization.
    """
    fc = model.fc if hasattr(model, 'fc') else model.module.fc
    if hasattr(fc, 'weight'):
        with torch.no_grad():
            w = fc.weight.data
            norms = torch.norm(w, dim=1, keepdim=True)
            normalized_w = w / (norms ** tau + 1e-12)
            fc.weight.data = normalized_w
    return model


# ========== Logging ==========

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ========== Misc ==========

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_class_split_info(cls_num_list, many_thr=100, few_thr=20,
                         dataset_name=None):
    """Get class split information for reporting."""
    many_classes, medium_classes, few_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=many_thr,
        few_thr=few_thr,
    )
    many = len(many_classes)
    medium = len(medium_classes)
    few = len(few_classes)
    return {
        'many': many, 'medium': medium, 'few': few,
        'total': len(cls_num_list),
        'max_count': max(cls_num_list),
        'min_count': min(cls_num_list),
        'imbalance_ratio': max(cls_num_list) / (min(cls_num_list) + 1e-12),
    }
