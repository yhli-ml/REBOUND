#!/usr/bin/env python3
"""Controlled Head2Tail counterfactual generation.

This is a conservative extension of ``generate_head2tail.py``. It keeps the
same output format but treats diffusion as a proposal mechanism:

  1. retrieve nearest head/source anchors,
  2. edit them toward a target class prompt,
  3. optionally suppress the source class in the negative prompt,
  4. accept only target-dominant outputs by CLIP text/prototype scores.

The goal is to reduce source-label leakage without changing the downstream
training loader.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from augment.head2tail_prompts import get_all_prompts, get_class_names, load_custom_prompts
from generate_head2tail import (
    build_nearest_source_cache,
    compute_augmentation_plan,
    compute_feature_prototypes,
    compute_per_image_plan,
    get_farthest_mapping,
    get_random_mapping,
    load_dataset_samples,
    select_head_source_sample,
)
from utils import get_class_split_lists


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, artifacts, distorted, watermark, text, deformed"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled Head2Tail counterfactual augmentation")

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10_lt', 'cifar100_lt', 'imagenet_lt'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--imb_factor', type=float, default=0.01)

    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--head_threshold', type=int, default=100)
    parser.add_argument('--tail_threshold', type=int, default=20)
    parser.add_argument('--feature_source', type=str, default='clip_image',
                        choices=['clip', 'clip_image'])
    parser.add_argument('--head_selection', type=str, default='nearest',
                        choices=['nearest', 'random', 'farthest', 'all'])
    parser.add_argument('--sample_selection', type=str, default='nearest',
                        choices=['nearest', 'random'])

    parser.add_argument('--model_id', type=str,
                        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--pipeline_type', type=str, default='img2img',
                        choices=['img2img', 'pix2pix'])
    parser.add_argument('--lora_weights', type=str, default='')
    parser.add_argument('--strength', type=float, default=0.6)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--gen_size', type=int, default=512)
    parser.add_argument('--n_prompts_per_class', type=int, default=5)
    parser.add_argument('--custom_prompts', type=str, default='')

    parser.add_argument('--plan_mode', type=str, default='uniformize',
                        choices=['per_image', 'target', 'uniformize'])
    parser.add_argument('--aug_per_image', type=int, default=1)
    parser.add_argument('--per_image_scope', type=str, default='medium_tail',
                        choices=['all', 'medium_tail', 'tail_only'])
    parser.add_argument('--target_num', type=int, default=-1,
                        help='Target count for target/uniformize modes. -1 = max class count.')
    parser.add_argument('--max_aug_per_class', type=int, default=500,
                        help='Cap per-class synthetic samples for target/uniformize modes.')
    parser.add_argument('--augment_medium', action='store_true',
                        help='Legacy target mode: include medium classes.')
    parser.add_argument('--head_strength', type=float, default=-1)

    parser.add_argument('--enable_filter', action='store_true',
                        help='Enable target-dominance acceptance filtering.')
    parser.add_argument('--target_text_threshold', type=float, default=-1.0,
                        help='Require CLIP image/text target score >= threshold. '
                             'Negative value disables this condition.')
    parser.add_argument('--prototype_margin_threshold', type=float, default=0.0,
                        help='Require target prototype score - source prototype score >= margin.')
    parser.add_argument('--source_negative_prompt', action='store_true',
                        help='Append source class name to the negative prompt.')
    parser.add_argument('--negative_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument('--keep_top_k_per_class', type=int, default=0,
                        help='After generation, keep only top K accepted samples per class. 0 = disabled.')
    parser.add_argument('--keep_ratio', type=float, default=1.0,
                        help='After generation, keep this ratio of accepted samples per class.')
    parser.add_argument('--max_attempt_multiplier', type=float, default=4.0,
                        help='Max attempts per requested sample when filtering is enabled.')

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32')

    args = parser.parse_args()
    if args.save_size == 0:
        args.save_size = 32 if args.dataset in ('cifar10_lt', 'cifar100_lt') else args.gen_size
    if args.aug_per_image <= 0:
        raise ValueError('--aug_per_image must be positive')
    if not (0 < args.keep_ratio <= 1.0):
        raise ValueError('--keep_ratio must be in (0, 1]')
    return args


def to_pil_image(sample):
    if isinstance(sample, np.ndarray):
        return Image.fromarray(sample).convert('RGB')
    if isinstance(sample, str):
        return Image.open(sample).convert('RGB')
    return sample.convert('RGB')


def safe_name(name):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)[:80]


def compute_uniformize_plan(cls_num_list, num_classes, target_num, max_aug_per_class,
                            head_threshold, tail_threshold, scope, dataset_name):
    head_classes, medium_classes, tail_classes = get_class_split_lists(
        cls_num_list,
        dataset_name=dataset_name,
        many_thr=head_threshold,
        few_thr=tail_threshold,
    )
    head_set = set(head_classes)
    medium_set = set(medium_classes)
    tail_set = set(tail_classes)

    if target_num <= 0:
        target_num = max(cls_num_list)

    aug_plan = {}
    head_plan = {}
    tail_plan = {}
    for cls_idx in range(num_classes):
        current = cls_num_list[cls_idx]
        if current <= 0 or current >= target_num:
            continue
        if scope == 'medium_tail' and cls_idx in head_set:
            continue
        if scope == 'tail_only' and cls_idx not in tail_set:
            continue

        n_to_gen = min(target_num - current, max_aug_per_class)
        if n_to_gen <= 0:
            continue
        aug_plan[cls_idx] = n_to_gen
        if scope == 'all' and cls_idx in head_set:
            head_plan[cls_idx] = n_to_gen
        else:
            tail_plan[cls_idx] = n_to_gen

    split = {
        'head': head_classes,
        'medium': medium_classes,
        'tail': tail_classes,
    }
    return aug_plan, head_plan, tail_plan, target_num, split


class ClipAcceptanceScorer:
    """CLIP target/source scorer for target-dominance filtering."""

    def __init__(self, class_names, prototype_tensor, device='cuda', clip_model='ViT-B/32'):
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "Install CLIP with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self.clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model.eval()
        self.class_names = class_names
        self.prototype_tensor = F.normalize(prototype_tensor.float(), dim=1).cpu()

        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        with torch.no_grad():
            text_features = []
            for start in range(0, len(prompts), 256):
                tokens = clip.tokenize(prompts[start:start + 256]).to(device)
                feats = self.model.encode_text(tokens).float()
                text_features.append(F.normalize(feats, dim=1).cpu())
        self.text_features = torch.cat(text_features, dim=0)

    @torch.no_grad()
    def encode_image(self, image):
        tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor).float()
        return F.normalize(feat, dim=1).cpu()[0]

    def score(self, image, target_class, source_class=None):
        feat = self.encode_image(image)
        target_text = float(torch.dot(feat, self.text_features[target_class]))
        text_top1 = int(torch.mv(self.text_features, feat).argmax().item())
        target_proto = float(torch.dot(feat, self.prototype_tensor[target_class]))
        proto_top1 = int(torch.mv(self.prototype_tensor, feat).argmax().item())

        out = {
            'target_text_score': target_text,
            'text_top1': text_top1,
            'target_proto_score': target_proto,
            'proto_top1': proto_top1,
            'acceptance_score': target_proto,
        }
        if source_class is not None:
            source_text = float(torch.dot(feat, self.text_features[source_class]))
            source_proto = float(torch.dot(feat, self.prototype_tensor[source_class]))
            out.update({
                'source_text_score': source_text,
                'text_target_minus_source': target_text - source_text,
                'source_proto_score': source_proto,
                'proto_target_minus_source': target_proto - source_proto,
                'acceptance_score': (target_proto - source_proto) + 0.1 * target_text,
            })
        return out

    def accepts(self, scores, target_text_threshold, prototype_margin_threshold):
        if target_text_threshold >= 0 and scores['target_text_score'] < target_text_threshold:
            return False, 'text_threshold'
        margin = scores.get('proto_target_minus_source')
        if margin is not None and margin < prototype_margin_threshold:
            return False, 'prototype_margin'
        return True, 'accepted'


def make_negative_prompt(base_prompt, source_name, enabled):
    if not enabled:
        return base_prompt
    source = source_name.replace('_', ' ')
    return f"{base_prompt}, {source}, a photo of a {source}"


def maybe_trim_metadata(metadata, keep_top_k, keep_ratio):
    if keep_top_k <= 0 and keep_ratio >= 1.0:
        return metadata, {}

    trim_stats = {}
    for class_key, entries in list(metadata.items()):
        if not entries:
            continue
        original_n = len(entries)
        keep_n = original_n
        if keep_ratio < 1.0:
            keep_n = min(keep_n, max(1, int(round(original_n * keep_ratio))))
        if keep_top_k > 0:
            keep_n = min(keep_n, keep_top_k)
        if keep_n < original_n:
            entries.sort(key=lambda e: e.get('acceptance_score', 0.0), reverse=True)
            metadata[class_key] = entries[:keep_n]
            trim_stats[class_key] = {'before': original_n, 'after': keep_n}
    return metadata, trim_stats


def save_outputs(args, metadata, config, mapping_path, stats):
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    txt_path = os.path.join(args.output_dir, 'augmented_list.txt')
    with open(txt_path, 'w') as f:
        for class_key in sorted(metadata.keys()):
            for entry in metadata[class_key]:
                f.write(f"{entry['path']} {entry['label']}\n")

    config['total_generated'] = sum(len(v) for v in metadata.values())
    config_path = os.path.join(args.output_dir, 'generation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    stats_path = os.path.join(args.output_dir, 'acceptance_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Metadata: {meta_path}")
    print(f"Image list: {txt_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Config: {config_path}")
    print(f"Acceptance stats: {stats_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    gen_size = (args.gen_size, args.gen_size)
    save_size = (args.save_size, args.save_size)
    head_strength = args.head_strength if args.head_strength >= 0 else args.strength

    print('=' * 70)
    print('Controlled Head2Tail Counterfactual Augmentation')
    print('=' * 70)
    print(f'Dataset: {args.dataset} (imb_factor={args.imb_factor})')
    print(f'Plan: {args.plan_mode}, scope={args.per_image_scope}, target_num={args.target_num}')
    print(f'Selection: head={args.head_selection}, source_sample={args.sample_selection}, top_k={args.top_k}')
    print(f'Filter: {args.enable_filter}, margin={args.prototype_margin_threshold}, text_thr={args.target_text_threshold}')
    print(f'Source-negative prompt: {args.source_negative_prompt}')
    print(f'Output: {args.output_dir}')

    class_samples, cls_num_list, num_classes, is_cifar, raw_ds = load_dataset_samples(args)
    class_names = get_class_names(args.dataset)
    print(f'Classes: {num_classes}, train samples: {sum(cls_num_list)}')

    from augment.head2tail_selector import HeadClassSelector
    selector = HeadClassSelector(args.dataset, device=args.device, feature_source=args.feature_source)
    include_medium_targets = args.plan_mode in ('per_image', 'uniformize') or args.augment_medium

    if args.head_selection == 'nearest':
        protos = compute_feature_prototypes(selector, args.feature_source, raw_ds)
        mapping, head_classes, target_classes = selector.get_head2tail_mapping(
            cls_num_list, top_k=args.top_k,
            head_threshold=args.head_threshold,
            tail_threshold=args.tail_threshold,
            include_medium_targets=include_medium_targets,
        )
    elif args.head_selection == 'random':
        mapping, head_classes, target_classes = get_random_mapping(
            num_classes, cls_num_list, args.head_threshold, args.tail_threshold,
            args.top_k, dataset_name=args.dataset,
            include_medium_targets=include_medium_targets,
        )
        protos = compute_feature_prototypes(selector, args.feature_source, raw_ds)
    elif args.head_selection == 'farthest':
        protos = compute_feature_prototypes(selector, args.feature_source, raw_ds)
        mapping, head_classes, target_classes = get_farthest_mapping(
            protos, cls_num_list, args.head_threshold, args.tail_threshold,
            args.top_k, dataset_name=args.dataset,
            include_medium_targets=include_medium_targets,
        )
    else:
        protos = compute_feature_prototypes(selector, args.feature_source, raw_ds)
        head_classes, medium_classes, tail_classes = get_class_split_lists(
            cls_num_list, dataset_name=args.dataset,
            many_thr=args.head_threshold, few_thr=args.tail_threshold,
        )
        target_classes = medium_classes + tail_classes if include_medium_targets else tail_classes
        mapping = {tc: [(hc, 0.0) for hc in head_classes] for tc in target_classes}

    mapping_path = os.path.join(args.output_dir, 'head2tail_mapping.json')
    selector.save_mapping(mapping, head_classes, target_classes, cls_num_list, mapping_path)

    if args.plan_mode == 'uniformize':
        aug_plan, head_aug_plan, tail_aug_plan, target_num, split = compute_uniformize_plan(
            cls_num_list, num_classes, args.target_num, args.max_aug_per_class,
            args.head_threshold, args.tail_threshold, args.per_image_scope, args.dataset,
        )
    elif args.plan_mode == 'per_image':
        aug_plan, head_aug_plan, tail_aug_plan = compute_per_image_plan(
            cls_num_list, num_classes, args.aug_per_image,
            args.head_threshold, args.tail_threshold, args.per_image_scope,
            dataset_name=args.dataset,
        )
        target_num = -1
        split = {}
    else:
        aug_plan, target_num = compute_augmentation_plan(
            cls_num_list, mapping, args.target_num, args.max_aug_per_class,
            args.augment_medium, args.head_threshold, args.tail_threshold,
            dataset_name=args.dataset,
        )
        head_aug_plan = {}
        tail_aug_plan = aug_plan
        split = {}

    total_requested = sum(aug_plan.values())
    print(f'Planned images: {total_requested}')
    print(f'Head self-aug classes: {len(head_aug_plan)} ({sum(head_aug_plan.values())} images)')
    print(f'Head2Tail target classes: {len(tail_aug_plan)} ({sum(tail_aug_plan.values())} images)')
    if total_requested == 0:
        print('Nothing to generate.')
        return

    if args.custom_prompts and os.path.exists(args.custom_prompts):
        all_prompts = load_custom_prompts(args.custom_prompts)
    else:
        all_prompts = get_all_prompts(args.dataset, args.n_prompts_per_class, args.seed)

    nearest_sample_cache = {}
    nearest_sample_cursors = defaultdict(int)
    if args.sample_selection == 'nearest' and selector.sample_features is not None:
        nearest_sample_cache = build_nearest_source_cache(selector, class_samples, mapping)
        print(f'Cached nearest source samples for {len(nearest_sample_cache)} target/source pairs')
    elif args.sample_selection == 'nearest':
        print('[WARN] nearest source selection requested but image sample features are unavailable; falling back to cycling')

    scorer = None
    if args.enable_filter:
        scorer = ClipAcceptanceScorer(class_names, protos, args.device, args.clip_model)

    from augment.head2tail_generator import Head2TailGenerator
    generator = Head2TailGenerator(
        model_id=args.model_id,
        pipeline_type=args.pipeline_type,
        device=args.device,
        lora_weights=args.lora_weights if args.lora_weights else None,
    )

    metadata = defaultdict(list)
    stats = {
        'requested_total': int(total_requested),
        'accepted_total': 0,
        'rejected_total': 0,
        'reject_reasons': Counter(),
        'per_class': {},
    }
    global_count = 0
    start_time = time.time()

    # Head classes remain self-augmentation proposals. We do not apply source
    # dominance filtering because source == target here.
    for head_c in sorted(head_aug_plan):
        class_key = f'class_{head_c:04d}'
        class_dir = os.path.join(args.output_dir, class_key)
        os.makedirs(class_dir, exist_ok=True)
        head_name = class_names[head_c] if head_c < len(class_names) else f'class_{head_c}'
        prompts = all_prompts.get(head_c, [f'a photo of a {head_name.replace("_", " ")}'])
        samples = class_samples.get(head_c, [])
        requested = head_aug_plan[head_c]
        accepted = 0
        attempts = 0
        max_attempts = max(requested, int(np.ceil(requested * args.max_attempt_multiplier)))
        print(f'[Self {head_c:04d}] {head_name}: requested={requested}')

        while accepted < requested and attempts < max_attempts and samples:
            sample_index = attempts % len(samples)
            src_pil = to_pil_image(samples[sample_index])
            prompt = prompts[attempts % len(prompts)]
            attempts += 1
            gen_img = generator.generate_with_retry(
                src_pil, prompt, strength=head_strength,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                size=gen_size, max_retries=2,
                num_inference_steps=args.num_inference_steps,
            )
            if gen_img is None:
                stats['reject_reasons']['invalid'] += 1
                continue
            if save_size != gen_size:
                gen_img = gen_img.resize(save_size, Image.LANCZOS)
            scores = scorer.score(gen_img, head_c, None) if scorer else {}
            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
            fname = f'self_{accepted:05d}_src{sample_index}_{prompt_hash}.jpg'
            rel_path = os.path.join(class_key, fname)
            gen_img.save(os.path.join(args.output_dir, rel_path), quality=95)
            entry = {
                'path': rel_path,
                'label': head_c,
                'source_class': head_c,
                'source_name': head_name,
                'similarity': 1.0,
                'prompt': prompt,
                'negative_prompt': args.negative_prompt,
                'strength': head_strength,
                'aug_type': 'self',
                'source_sample_index': sample_index,
                'source_sample_rank': 0,
                'source_sample_strategy': 'self',
                **scores,
            }
            metadata[class_key].append(entry)
            accepted += 1
            global_count += 1

        stats['per_class'][str(head_c)] = {
            'requested': int(requested), 'accepted': int(accepted),
            'attempts': int(attempts), 'mode': 'self'
        }

    for tail_c in sorted(tail_aug_plan):
        class_key = f'class_{tail_c:04d}'
        class_dir = os.path.join(args.output_dir, class_key)
        os.makedirs(class_dir, exist_ok=True)
        tail_name = class_names[tail_c] if tail_c < len(class_names) else f'class_{tail_c}'
        head_sources = mapping.get(tail_c, [])
        prompts = all_prompts.get(tail_c, [f'a photo of a {tail_name.replace("_", " ")}'])
        requested = tail_aug_plan[tail_c]
        accepted = 0
        attempts = 0
        max_attempts = max(requested, int(np.ceil(requested * args.max_attempt_multiplier)))
        reject_reasons = Counter()
        print(f'[H2T {tail_c:04d}] {tail_name}: requested={requested}, sources={len(head_sources)}')

        while accepted < requested and attempts < max_attempts and head_sources:
            source_idx = attempts % len(head_sources)
            head_c, similarity = head_sources[source_idx]
            head_samples = class_samples.get(head_c, [])
            if not head_samples:
                attempts += 1
                continue

            sample, source_sample_index, source_sample_rank, source_sample_strategy = select_head_source_sample(
                head_samples=head_samples,
                tail_class=tail_c,
                head_class=head_c,
                n_generated=attempts,
                sample_selection=args.sample_selection,
                nearest_sample_cache=nearest_sample_cache,
                nearest_sample_cursors=nearest_sample_cursors,
            )
            head_pil = to_pil_image(sample)
            prompt = prompts[attempts % len(prompts)]
            head_name = class_names[head_c] if head_c < len(class_names) else f'class_{head_c}'
            negative_prompt = make_negative_prompt(
                args.negative_prompt, head_name, args.source_negative_prompt
            )
            attempts += 1

            gen_img = generator.generate_with_retry(
                head_pil, prompt, strength=args.strength,
                guidance_scale=args.guidance_scale,
                negative_prompt=negative_prompt,
                size=gen_size, max_retries=2,
                num_inference_steps=args.num_inference_steps,
            )
            if gen_img is None:
                reject_reasons['invalid'] += 1
                stats['reject_reasons']['invalid'] += 1
                continue
            if save_size != gen_size:
                gen_img_for_score = gen_img.resize(save_size, Image.LANCZOS)
            else:
                gen_img_for_score = gen_img

            scores = scorer.score(gen_img_for_score, tail_c, head_c) if scorer else {}
            if scorer:
                ok, reason = scorer.accepts(
                    scores,
                    args.target_text_threshold,
                    args.prototype_margin_threshold,
                )
                if not ok:
                    reject_reasons[reason] += 1
                    stats['reject_reasons'][reason] += 1
                    stats['rejected_total'] += 1
                    continue

            gen_to_save = gen_img_for_score
            prompt_hash = hashlib.md5(f'{prompt}|{negative_prompt}'.encode('utf-8')).hexdigest()[:8]
            fname = f'h2t_{accepted:05d}_from{head_c}_{safe_name(head_name)}_{prompt_hash}.jpg'
            rel_path = os.path.join(class_key, fname)
            gen_to_save.save(os.path.join(args.output_dir, rel_path), quality=95)
            entry = {
                'path': rel_path,
                'label': tail_c,
                'source_class': head_c,
                'source_name': head_name,
                'similarity': round(float(similarity), 4),
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'strength': args.strength,
                'aug_type': 'head2tail_controlled',
                'source_sample_index': source_sample_index,
                'source_sample_rank': source_sample_rank,
                'source_sample_strategy': source_sample_strategy,
                **scores,
            }
            metadata[class_key].append(entry)
            accepted += 1
            global_count += 1

            if global_count % 50 == 0:
                elapsed = time.time() - start_time
                speed = global_count / max(elapsed, 1e-6)
                print(f'  Progress: accepted={global_count}/{total_requested}, speed={speed:.2f} img/s')

        stats['per_class'][str(tail_c)] = {
            'requested': int(requested),
            'accepted': int(accepted),
            'attempts': int(attempts),
            'mode': 'head2tail',
            'reject_reasons': dict(reject_reasons),
        }
        if accepted < requested:
            print(f'  [WARN] class {tail_c} accepted {accepted}/{requested} after {attempts} attempts')

    metadata, trim_stats = maybe_trim_metadata(metadata, args.keep_top_k_per_class, args.keep_ratio)
    stats['trim_stats'] = trim_stats
    stats['accepted_total'] = int(sum(len(v) for v in metadata.values()))
    stats['reject_reasons'] = dict(stats['reject_reasons'])

    config = {
        'method': 'head2tail_controlled',
        'dataset': args.dataset,
        'imb_factor': args.imb_factor,
        'plan_mode': args.plan_mode,
        'per_image_scope': args.per_image_scope,
        'target_num': target_num,
        'max_aug_per_class': args.max_aug_per_class,
        'head_selection': args.head_selection,
        'sample_selection': args.sample_selection,
        'top_k': args.top_k,
        'source_negative_prompt': args.source_negative_prompt,
        'enable_filter': args.enable_filter,
        'target_text_threshold': args.target_text_threshold,
        'prototype_margin_threshold': args.prototype_margin_threshold,
        'keep_top_k_per_class': args.keep_top_k_per_class,
        'keep_ratio': args.keep_ratio,
        'strength': args.strength,
        'head_strength': head_strength,
        'guidance_scale': args.guidance_scale,
        'model_id': args.model_id,
        'pipeline_type': args.pipeline_type,
        'lora_weights': args.lora_weights,
        'aug_per_image': args.aug_per_image if args.plan_mode == 'per_image' else None,
        'gen_size': args.gen_size,
        'save_size': args.save_size,
        'seed': args.seed,
    }
    save_outputs(args, dict(metadata), config, mapping_path, stats)

    elapsed = time.time() - start_time
    print('=' * 70)
    print(f'Done. Accepted {stats["accepted_total"]}/{total_requested} requested images in {elapsed / 60:.1f} min')
    print(f'Reject reasons: {stats["reject_reasons"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
