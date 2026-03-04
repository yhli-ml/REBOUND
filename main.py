"""
Long-Tail Learning Baseline
============================

A unified framework for long-tailed visual recognition benchmarking.

Supported Methods:
  - ERM (vanilla cross-entropy)
#   - Class-Balanced CE (Cui et al., CVPR 2019)
#   - Focal Loss (Lin et al., ICCV 2017)
#   - LDAM-DRW (Cao et al., NeurIPS 2019)
#   - Logit Adjustment (Menon et al., ICLR 2021)
#   - Balanced Softmax (Ren et al., NeurIPS 2020)
#   - Seesaw Loss (Wang et al., CVPR 2021)
#   - Mixup / CutMix / Remix
#   - cRT (Kang et al., ICLR 2020) - two-stage
#   - tau-Normalization (Kang et al., ICLR 2020) - post-hoc

Supported Backbones:
  - ResNet-32 (CIFAR)
  - ResNet-50/101/152 (ImageNet-scale)
  - ResNeXt-50-32x4d

Supported Datasets:
  - CIFAR-10-LT, CIFAR-100-LT
  - ImageNet-LT
  - iNaturalist 2018
  - Places-LT

Usage:
  python main.py --dataset cifar100_lt --arch resnet32 --loss ldam --drw 160 --epochs 200
  python main.py --dataset imagenet_lt --arch resnet50 --loss balanced_softmax --epochs 100
  python main.py --dataset cifar100_lt --arch resnet32 --loss ce --mixup --epochs 200
  python main.py --dataset imagenet_lt --arch resnet50 --loss ce --stage2 crt --epochs 100
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataset
from datasets.cifar_lt import get_cifar_train_transform, get_cifar_test_transform
from datasets.imagenet_lt import get_imagenet_train_transform, get_imagenet_test_transform
from models import create_model
from losses import get_loss
from losses.ldam_loss import get_drw_weights
from losses.mixup import mixup_data, mixup_criterion, cutmix_data, remix_data
from losses.logit_adjust import logit_adjust_posthoc
from samplers import ClassAwareSampler, ClassBalancedSampler
from datasets.diffusemix_dataset import DiffuseMixDataset
from utils import (
    shot_acc, accuracy, AverageMeter,
    adjust_learning_rate, save_checkpoint, load_checkpoint,
    tau_normalize, setup_logger, get_class_split_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Long-Tail Learning Baseline')

    # ===== Dataset =====
    parser.add_argument('--dataset', type=str, default='cifar100_lt',
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt',
                                 'inaturalist', 'places_lt'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset annotations')
    parser.add_argument('--img_root', type=str, default='',
                        help='Root directory for images (if different from data_root). '
                             'For ImageNet-LT: the directory containing train/ and val/ folders')
    parser.add_argument('--imb_factor', type=float, default=0.01,
                        help='Imbalance factor for CIFAR (1/IF). 0.01=100x imbalance')
    parser.add_argument('--augment', type=str, default='standard',
                        choices=['standard', 'autoaug', 'randaug'],
                        help='Data augmentation strategy')

    # ===== Model =====
    parser.add_argument('--arch', type=str, default='resnet32',
                        choices=['resnet20', 'resnet32', 'resnet44', 'resnet56',
                                 'resnet10', 'resnet50', 'resnet101', 'resnet152',
                                 'resnext50'],
                        help='Backbone architecture')
    parser.add_argument('--use_norm', action='store_true',
                        help='Use cosine classifier (NormedLinear)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained backbone')

    # ===== Loss =====
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'ce_weighted', 'cb_ce', 'focal', 'ldam',
                                 'logit_adjust', 'la', 'balanced_softmax', 'bs',
                                 'seesaw', 'ride'],
                        help='Loss function')

    # Loss-specific hyperparameters
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--cb_beta', type=float, default=0.9999,
                        help='Class-balanced loss beta')
    parser.add_argument('--ldam_max_m', type=float, default=0.5,
                        help='LDAM max margin')
    parser.add_argument('--ldam_s', type=float, default=30.0,
                        help='LDAM scaling factor')
    parser.add_argument('--la_tau', type=float, default=1.0,
                        help='Logit Adjustment tau')
    parser.add_argument('--seesaw_p', type=float, default=0.8,
                        help='Seesaw mitigation exponent')
    parser.add_argument('--seesaw_q', type=float, default=2.0,
                        help='Seesaw compensation exponent')

    # ===== DRW (Deferred Re-Weighting) =====
    parser.add_argument('--drw', type=int, default=-1,
                        help='Epoch to start DRW. -1 = disabled. '
                             'Typical: 160 for 200 epochs')

    # ===== Mixup / CutMix / Remix =====
    parser.add_argument('--mixup', action='store_true', help='Enable Mixup')
    parser.add_argument('--cutmix', action='store_true', help='Enable CutMix')
    parser.add_argument('--remix', action='store_true', help='Enable Remix')
    parser.add_argument('--mix_alpha', type=float, default=1.0,
                        help='Mixup/CutMix/Remix alpha parameter')

    # ===== DiffuseMix Augmentation =====
    parser.add_argument('--diffusemix_dir', type=str, default='',
                        help='Directory with pre-generated DiffuseMix images. '
                             'If set, augmented images are added to training set.')
    parser.add_argument('--diffusemix_ratio', type=float, default=1.0,
                        help='Fraction of augmented data to use (0.0~1.0). '
                             'E.g. 0.25 = use only 25%% of augmented images.')
    parser.add_argument('--diffusemix_weight', type=float, default=1.0,
                        help='Loss weight for augmented samples (0.0~1.0). '
                             '1.0 = same weight as original, 0.5 = half weight.')
    parser.add_argument('--use_orig_cls_num', action='store_true',
                        help='Use original dataset class counts for loss/eval '
                             '(not augmented counts). Important for fair comparison.') 

    # ===== Two-Stage Methods =====
    parser.add_argument('--stage2', type=str, default='none',
                        choices=['none', 'crt', 'tau_norm', 'la_posthoc'],
                        help='Second stage method: '
                             'crt = classifier retraining, '
                             'tau_norm = tau-normalization, '
                             'la_posthoc = post-hoc logit adjustment')
    parser.add_argument('--stage2_epochs', type=int, default=10,
                        help='Number of epochs for cRT stage 2')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Tau for tau-normalization or post-hoc LA')
    parser.add_argument('--stage1_ckpt', type=str, default='',
                        help='Checkpoint from stage 1 (for cRT/tau-norm)')

    # ===== Sampling =====
    parser.add_argument('--sampler', type=str, default='none',
                        choices=['none', 'class_aware', 'class_balanced', 'square_root'],
                        help='Sampler for training data loader')

    # ===== Training =====
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr_schedule', type=str, default='warmup_step',
                        choices=['cosine', 'step', 'warmup_step', 'warmup_cosine', 'linear'],
                        help='Learning rate schedule')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[160, 180],
                        help='LR decay milestones (for step schedule)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='LR decay factor (for step schedule)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (for warmup_cosine schedule)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')

    # ===== System =====
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment name (auto-generated if empty)')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only (no training)')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency (iterations)')

    args = parser.parse_args()

    # Auto-generate experiment name
    if not args.exp_name:
        parts = [args.dataset, args.arch, args.loss]
        if args.dataset in ('cifar10_lt', 'cifar100_lt'):
            parts.append(f'IF{int(1/args.imb_factor)}')
        if args.use_norm:
            parts.append('norm')
        if args.drw >= 0:
            parts.append(f'drw{args.drw}')
        if args.mixup:
            parts.append('mixup')
        if args.cutmix:
            parts.append('cutmix')
        if args.remix:
            parts.append('remix')
        if args.sampler != 'none':
            parts.append(args.sampler)
        if args.diffusemix_dir:
            parts.append('diffusemix')
        if args.stage2 != 'none':
            parts.append(args.stage2)
        args.exp_name = '_'.join(parts)

    args.save_dir = os.path.join(args.output_dir, args.exp_name)
    return args


def get_num_classes(dataset_name):
    """Return number of classes for each dataset."""
    return {
        'cifar10_lt': 10,
        'cifar100_lt': 100,
        'imagenet_lt': 1000,
        'inaturalist': 8142,
        'places_lt': 365,
    }[dataset_name]


def build_datasets(args):
    """Build train and val datasets with appropriate transforms."""
    is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')

    if is_cifar:
        train_transform = get_cifar_train_transform(args.augment)
        val_transform = get_cifar_test_transform()
    else:
        train_transform = get_imagenet_train_transform(args.augment)
        val_transform = get_imagenet_test_transform()

    # For ImageNet-LT etc., annotations and images may be in different dirs
    img_root = args.img_root if args.img_root else None

    train_dataset = get_dataset(
        args.dataset, args.data_root, train=True,
        transform=train_transform, imb_factor=args.imb_factor,
        download=True, img_root=img_root)
    val_dataset = get_dataset(
        args.dataset, args.data_root, train=False,
        transform=val_transform, imb_factor=args.imb_factor,
        download=True, img_root=img_root)

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, args):
    """Build data loaders with optional balanced sampling."""
    # Training sampler
    train_sampler = None
    shuffle = True

    if args.sampler == 'class_aware':
        train_sampler = ClassAwareSampler(train_dataset)
        shuffle = False
    elif args.sampler == 'class_balanced':
        train_sampler = ClassBalancedSampler(train_dataset)
        shuffle = False
    elif args.sampler == 'square_root':
        # Square-root resampling
        targets = np.array(train_dataset.targets)
        cls_counts = np.bincount(targets, minlength=get_num_classes(args.dataset))
        sqrt_counts = np.sqrt(cls_counts)
        per_cls_weight = sqrt_counts / sqrt_counts.sum()
        sample_weights = per_cls_weight[targets]
        sample_weights = torch.from_numpy(sample_weights).double()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(train_dataset), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, args, logger):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    cls_num_list = train_loader.dataset.cls_num_list
    use_diffusemix_weight = (hasattr(args, 'diffusemix_weight')
                             and args.diffusemix_weight < 1.0
                             and hasattr(args, 'diffusemix_dir')
                             and args.diffusemix_dir)

    for i, batch in enumerate(train_loader):
        # Handle both 2-tuple (img, label) and 3-tuple (img, label, is_aug)
        if len(batch) == 3:
            images, targets, is_aug = batch
            is_aug = is_aug.cuda(args.gpu, non_blocking=True)
        else:
            images, targets = batch
            is_aug = None

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # Data augmentation: Mixup / CutMix / Remix
        use_mix = args.mixup or args.cutmix or args.remix
        if use_mix:
            if args.remix:
                images, targets_a, targets_b, lam = remix_data(
                    images, targets, cls_num_list, alpha=args.mix_alpha)
            elif args.cutmix:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, alpha=args.mix_alpha)
            else:  # mixup
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, alpha=args.mix_alpha)

        # Forward
        logits = model(images)

        # Loss
        if use_mix:
            if isinstance(lam, torch.Tensor):
                # Remix returns per-sample lambda
                loss = (lam * nn.functional.cross_entropy(logits, targets_a, reduction='none') +
                        (1 - lam) * nn.functional.cross_entropy(logits, targets_b, reduction='none')).mean()
            else:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        else:
            loss = criterion(logits, targets)

        # Apply per-sample loss weighting for DiffuseMix augmented samples
        if use_diffusemix_weight and is_aug is not None and not use_mix:
            per_sample_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
            weights = torch.where(is_aug == 1,
                                  torch.tensor(args.diffusemix_weight, device=images.device),
                                  torch.tensor(1.0, device=images.device))
            loss = (per_sample_loss * weights).mean()

        # Metrics
        acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, logits.size(1))))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            logger.info(
                f'Epoch [{epoch}][{i+1}/{len(train_loader)}] '
                f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                f'Acc@1 {top1.val:.2f} ({top1.avg:.2f}) '
                f'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'
            )

    return losses.avg, top1.avg


@torch.no_grad()
def validate(model, val_loader, criterion, args, cls_num_list, logger,
             posthoc_la=False, posthoc_tau=0.0):
    """Validate the model."""
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    all_preds = []
    all_targets = []

    for images, targets in val_loader:
        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        logits = model(images)

        # Post-hoc logit adjustment
        if posthoc_la:
            logits = logit_adjust_posthoc(logits, cls_num_list, tau=posthoc_tau)

        loss = criterion(logits, targets) if criterion is not None else 0.0
        acc1, _ = accuracy(logits, targets, topk=(1, min(5, logits.size(1))))

        if criterion is not None:
            losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        _, preds = logits.max(1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute Many/Medium/Few shot accuracy
    if cls_num_list is not None:
        shot_results = shot_acc(all_preds, all_targets, cls_num_list)
        logger.info(
            f'[Val] Loss: {losses.avg:.4f} | '
            f'Overall: {shot_results["overall"]:.2f}% | '
            f'Many: {shot_results["many"]:.2f}% | '
            f'Medium: {shot_results["medium"]:.2f}% | '
            f'Few: {shot_results["few"]:.2f}%'
        )
    else:
        shot_results = {'overall': top1.avg, 'many': 0, 'medium': 0, 'few': 0}
        logger.info(f'[Val] Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f}%')

    return shot_results


def run_stage2(model, train_dataset, val_loader, args, cls_num_list, logger, writer):
    """Run second-stage methods (cRT, tau-norm, post-hoc LA)."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Stage 2: {args.stage2}')
    logger.info(f'{"="*60}')

    if args.stage2 == 'tau_norm':
        # Tau-normalization: just adjust the classifier weights
        logger.info(f'Applying tau-normalization with tau={args.tau}')
        tau_normalize(model, tau=args.tau)
        result = validate(model, val_loader, None, args, cls_num_list, logger)
        return result

    elif args.stage2 == 'la_posthoc':
        # Post-hoc logit adjustment
        logger.info(f'Applying post-hoc logit adjustment with tau={args.tau}')
        result = validate(model, val_loader, None, args, cls_num_list, logger,
                          posthoc_la=True, posthoc_tau=args.tau)
        return result

    elif args.stage2 == 'crt':
        # Classifier Re-Training (cRT)
        logger.info('Freezing backbone, retraining classifier with balanced sampling')

        # Freeze backbone
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        # Reset classifier
        num_classes = get_num_classes(args.dataset)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes, use_norm=args.use_norm)
        model.cuda(args.gpu)

        # Balanced sampler for stage 2
        is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')
        if is_cifar:
            transform = get_cifar_train_transform(args.augment)
        else:
            transform = get_imagenet_train_transform(args.augment)
        train_dataset.transform = transform

        crt_sampler = ClassAwareSampler(train_dataset)
        crt_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=crt_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=True)

        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 0.1, momentum=args.momentum,
            weight_decay=args.weight_decay)

        best_result = None
        for epoch in range(args.stage2_epochs):
            model.train()
            losses = AverageMeter()
            for batch in crt_loader:
                if len(batch) == 3:
                    images, targets, _ = batch
                else:
                    images, targets = batch
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), images.size(0))

            logger.info(f'[cRT Epoch {epoch+1}/{args.stage2_epochs}] Loss: {losses.avg:.4f}')
            result = validate(model, val_loader, criterion, args, cls_num_list, logger)

            if best_result is None or result['overall'] > best_result['overall']:
                best_result = result
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': result['overall'],
                }, True, args.save_dir, 'checkpoint_crt.pth')

            if writer:
                writer.add_scalar('crt/loss', losses.avg, epoch)
                writer.add_scalar('crt/acc_overall', result['overall'], epoch)

        # Unfreeze
        for param in model.parameters():
            param.requires_grad = True

        return best_result


def main():
    args = parse_args()

    # ===== Setup =====
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger('LT_baseline', os.path.join(args.save_dir, 'train.log'))
    logger.info(f'Arguments: {args}')

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Device
    torch.cuda.set_device(args.gpu)

    # ===== Data =====
    train_dataset, val_dataset = build_datasets(args)

    # Wrap with DiffuseMix augmented data if specified
    orig_cls_num_list = list(train_dataset.cls_num_list)  # Save original counts
    if args.diffusemix_dir and os.path.isdir(args.diffusemix_dir):
        is_cifar = args.dataset in ('cifar10_lt', 'cifar100_lt')
        aug_img_size = 32 if is_cifar else None
        train_dataset = DiffuseMixDataset(
            original_dataset=train_dataset,
            diffusemix_dir=args.diffusemix_dir,
            transform=train_dataset.transform,
            is_cifar=is_cifar,
            aug_img_size=aug_img_size,
            sample_ratio=args.diffusemix_ratio,
        )
        logger.info(f'DiffuseMix: loaded augmented data from {args.diffusemix_dir}')
        logger.info(f'DiffuseMix: ratio={args.diffusemix_ratio}, '
                    f'weight={args.diffusemix_weight}')
    elif args.diffusemix_dir:
        logger.warning(f'DiffuseMix dir not found: {args.diffusemix_dir}, skipping.')

    # Use original class counts for loss/eval if requested
    if args.use_orig_cls_num:
        cls_num_list = orig_cls_num_list
        logger.info(f'Using ORIGINAL cls_num_list for loss/eval '
                    f'(min={min(cls_num_list)}, max={max(cls_num_list)})')
    else:
        cls_num_list = train_dataset.cls_num_list
    num_classes = get_num_classes(args.dataset)

    split_info = get_class_split_info(cls_num_list)
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'  Classes: {num_classes} (Many: {split_info["many"]}, '
                f'Medium: {split_info["medium"]}, Few: {split_info["few"]})')
    logger.info(f'  Max count: {split_info["max_count"]}, '
                f'Min count: {split_info["min_count"]}, '
                f'Imbalance ratio: {split_info["imbalance_ratio"]:.1f}')
    logger.info(f'  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, args)

    # ===== Model =====
    model = create_model(args.arch, num_classes, use_norm=args.use_norm,
                         pretrained=args.pretrained)
    model = model.cuda(args.gpu)
    logger.info(f'Model: {args.arch} (feat_dim={model.feat_dim}, '
                f'use_norm={args.use_norm}, pretrained={args.pretrained})')

    # ===== Loss =====
    criterion = get_loss(args.loss, cls_num_list, args)
    criterion = criterion.cuda(args.gpu)
    logger.info(f'Loss: {args.loss}')

    # ===== Optimizer =====
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    # ===== TensorBoard =====
    writer = SummaryWriter(os.path.join(args.save_dir, 'tb'))

    # ===== Resume =====
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        info = load_checkpoint(model, args.resume, optimizer)
        start_epoch = info.get('epoch', 0) + 1
        best_acc = info.get('best_acc', 0.0)

    if args.stage1_ckpt and args.stage2 != 'none':
        logger.info(f'Loading stage 1 checkpoint: {args.stage1_ckpt}')
        load_checkpoint(model, args.stage1_ckpt, strict=True)

    # ===== Evaluate only =====
    if args.evaluate:
        validate(model, val_loader, criterion, args, cls_num_list, logger)
        if args.stage2 != 'none':
            run_stage2(model, train_dataset, val_loader, args,
                       cls_num_list, logger, writer)
        return

    # ===== Training Loop =====
    logger.info(f'\nStarting training for {args.epochs} epochs...\n')

    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)

        # DRW: activate class weights at specified epoch
        if args.drw >= 0 and epoch >= args.drw:
            drw_weights = get_drw_weights(cls_num_list).cuda(args.gpu)
            if hasattr(criterion, 'update_weight'):
                criterion.update_weight(drw_weights)
            elif hasattr(criterion, 'weight'):
                criterion.weight = drw_weights

        # RIDE-specific epoch update
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args, logger)

        # Validate
        result = validate(model, val_loader, criterion, args, cls_num_list, logger)

        # Save
        is_best = result['overall'] > best_acc
        best_acc = max(result['overall'], best_acc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': vars(args),
        }, is_best, args.save_dir)

        # TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.add_scalar('val/overall', result['overall'], epoch)
        writer.add_scalar('val/many', result['many'], epoch)
        writer.add_scalar('val/medium', result['medium'], epoch)
        writer.add_scalar('val/few', result['few'], epoch)

        logger.info(f'Epoch {epoch}: Best Acc = {best_acc:.2f}%\n')

    logger.info(f'\nTraining finished. Best accuracy: {best_acc:.2f}%')

    # ===== Stage 2 =====
    if args.stage2 != 'none':
        # Load best model from stage 1
        load_checkpoint(model, os.path.join(args.save_dir, 'best_model.pth'))
        result = run_stage2(model, train_dataset, val_loader, args,
                            cls_num_list, logger, writer)
        if result:
            logger.info(f'Stage 2 best accuracy: {result["overall"]:.2f}%')

    writer.close()


if __name__ == '__main__':
    main()
