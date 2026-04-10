#!/usr/bin/env python3
"""Evaluate semantic faithfulness and source-label leakage in generated data.

This script is intentionally method-agnostic: it can compare Head2Tail,
BarePrompt, DiffuseMix, or any generator that writes the shared repository
format:

  generated_dir/
    metadata.json
    class_XXXX/*.jpg

The core diagnostic is target-vs-source alignment. Head2Tail images have a
target label and often a different source class. If the generated image is
closer to the source class prototype/text than to the target, that is evidence
of source-label leakage and ambiguous supervision.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP-based semantic diagnostics for generated images")
    parser.add_argument("--generated_dirs", nargs="+", required=True,
                        help="Generated image directories to evaluate")
    parser.add_argument("--method_names", nargs="*", default=[],
                        help="Optional display names, one per generated dir")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10_lt", "cifar100_lt", "imagenet_lt"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--img_root", type=str, default="",
                        help="Image root for ImageNet-LT")
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./eval_generated_semantics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_gen_per_method", type=int, default=8000,
                        help="0 = evaluate every generated image")
    parser.add_argument("--max_real_per_class", type=int, default=80,
                        help="Max real train images per class for prototypes")
    parser.add_argument("--max_grid_images", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_real_prototypes", action="store_true",
                        help="Only use CLIP text prompts, not real-image prototypes")
    return parser.parse_args()


def load_clip(name, device):
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "The `clip` package is required. Install with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc
    model, preprocess = clip.load(name, device=device)
    model.eval()
    return clip, model, preprocess


def get_num_classes(dataset):
    return {"cifar10_lt": 10, "cifar100_lt": 100, "imagenet_lt": 1000}[dataset]


def get_image_for_dataset_sample(dataset, idx):
    """Return an untransformed PIL image from a repository dataset instance."""
    if hasattr(dataset, "data"):
        return Image.fromarray(dataset.data[idx]).convert("RGB")
    if hasattr(dataset, "img_paths"):
        return Image.open(dataset.img_paths[idx]).convert("RGB")
    img, _ = dataset[idx]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    raise TypeError("Cannot recover PIL image from dataset sample")


def load_real_class_images(args):
    from datasets import get_dataset

    dataset = get_dataset(
        args.dataset,
        args.data_root,
        train=True,
        transform=None,
        imb_factor=args.imb_factor,
        download=True,
        img_root=args.img_root if args.img_root else None,
    )

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_indices[int(label)].append(idx)

    rng = random.Random(args.seed)
    class_images = {}
    for cls, indices in class_to_indices.items():
        picked = list(indices)
        if len(picked) > args.max_real_per_class:
            picked = rng.sample(picked, args.max_real_per_class)
        class_images[cls] = [get_image_for_dataset_sample(dataset, i) for i in picked]
    return class_images, list(dataset.cls_num_list)


def flatten_metadata(generated_dir):
    generated_dir = Path(generated_dir)
    meta_path = generated_dir / "metadata.json"
    entries = []

    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        for class_key, class_entries in metadata.items():
            if not isinstance(class_entries, list):
                continue
            fallback_label = int(class_key.split("_")[1])
            for entry in class_entries:
                item = dict(entry)
                item.setdefault("label", fallback_label)
                item["abs_path"] = str(generated_dir / item["path"])
                entries.append(item)
    else:
        for class_dir in sorted(generated_dir.glob("class_*")):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name.split("_")[1])
            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    entries.append({
                        "path": str(path.relative_to(generated_dir)),
                        "abs_path": str(path),
                        "label": label,
                    })

    return [e for e in entries if os.path.exists(e["abs_path"])]


@torch.no_grad()
def encode_pil_images(images, clip_model, preprocess, device, batch_size):
    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        feat = clip_model.encode_image(tensors).float()
        feats.append(F.normalize(feat, dim=1).cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0, 512)


@torch.no_grad()
def encode_text_prompts(class_names, clip_module, clip_model, device, batch_size):
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    feats = []
    for start in range(0, len(prompts), batch_size):
        tokens = clip_module.tokenize(prompts[start:start + batch_size]).to(device)
        feat = clip_model.encode_text(tokens).float()
        feats.append(F.normalize(feat, dim=1).cpu())
    return torch.cat(feats, dim=0), prompts


def compute_real_prototypes(real_images, clip_model, preprocess, device, batch_size,
                            num_classes):
    proto = torch.zeros(num_classes, clip_model.visual.output_dim)
    present = torch.zeros(num_classes, dtype=torch.bool)
    for cls in sorted(real_images):
        images = real_images[cls]
        if not images:
            continue
        feats = encode_pil_images(images, clip_model, preprocess, device, batch_size)
        proto[cls] = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)[0]
        present[cls] = True
    return proto, present


def sample_entries(entries, max_count, seed):
    if max_count <= 0 or len(entries) <= max_count:
        return entries
    rng = random.Random(seed)
    return rng.sample(entries, max_count)


def score_entries(entries, method, class_names, text_feats, proto_feats, proto_present,
                  clip_model, preprocess, device, batch_size):
    use_proto = proto_feats is not None and proto_present is not None

    rows = []
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start:start + batch_size]
        images = [
            Image.open(e["abs_path"]).convert("RGB")
            for e in batch_entries
        ]
        image_feats = encode_pil_images(
            images, clip_model, preprocess, device, batch_size
        )

        text_logits = image_feats @ text_feats.t()
        text_pred = text_logits.argmax(dim=1)

        if use_proto:
            proto_logits = image_feats @ proto_feats.t()
            proto_logits[:, ~proto_present] = -1e6
            proto_pred = proto_logits.argmax(dim=1)
        else:
            proto_logits = None
            proto_pred = None

        for idx, entry in enumerate(batch_entries):
            label = int(entry["label"])
            source = entry.get("source_class")
            source = int(source) if source is not None else None

            target_text_score = float(text_logits[idx, label])
            source_text_score = (
                float(text_logits[idx, source])
                if source is not None and 0 <= source < text_logits.shape[1]
                else None
            )
            text_margin = (
                target_text_score - source_text_score
                if source_text_score is not None
                else None
            )

            if use_proto:
                target_proto_score = float(proto_logits[idx, label])
                source_proto_score = (
                    float(proto_logits[idx, source])
                    if source is not None and 0 <= source < proto_logits.shape[1]
                    and bool(proto_present[source])
                    else None
                )
                proto_margin = (
                    target_proto_score - source_proto_score
                    if source_proto_score is not None
                    else None
                )
                proto_top1 = int(proto_pred[idx])
            else:
                target_proto_score = None
                source_proto_score = None
                proto_margin = None
                proto_top1 = None

            rows.append({
                "method": method,
                "path": entry["abs_path"],
                "rel_path": entry.get("path", ""),
                "label": label,
                "target_name": class_names[label] if label < len(class_names) else str(label),
                "source_class": source,
                "source_name": (
                    class_names[source] if source is not None and source < len(class_names)
                    else ""
                ),
                "aug_type": entry.get("aug_type", ""),
                "prompt": entry.get("prompt", ""),
                "text_target_score": target_text_score,
                "text_source_score": source_text_score,
                "text_target_minus_source": text_margin,
                "text_top1": int(text_pred[idx]),
                "text_top1_name": class_names[int(text_pred[idx])],
                "text_top1_is_target": int(int(text_pred[idx]) == label),
                "text_top1_is_source": int(source is not None and int(text_pred[idx]) == source),
                "proto_target_score": target_proto_score,
                "proto_source_score": source_proto_score,
                "proto_target_minus_source": proto_margin,
                "proto_top1": proto_top1,
                "proto_top1_name": class_names[proto_top1] if proto_top1 is not None else "",
                "proto_top1_is_target": (
                    int(proto_top1 == label) if proto_top1 is not None else ""
                ),
                "proto_top1_is_source": (
                    int(source is not None and proto_top1 == source)
                    if proto_top1 is not None else ""
                ),
                "source_leak_text": (
                    int(source_text_score is not None and source_text_score > target_text_score)
                ),
                "source_leak_proto": (
                    int(source_proto_score is not None and source_proto_score > target_proto_score)
                ),
            })
    return rows


def summarize_rows(rows):
    if not rows:
        return {}

    def mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None

    def rate(key):
        vals = [r[key] for r in rows if r.get(key) != ""]
        return float(np.mean(vals)) if vals else None

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    return {
        "n": len(rows),
        "n_classes": len(by_label),
        "text_top1_target_rate": rate("text_top1_is_target"),
        "text_top1_source_rate": rate("text_top1_is_source"),
        "proto_top1_target_rate": rate("proto_top1_is_target"),
        "proto_top1_source_rate": rate("proto_top1_is_source"),
        "source_leak_text_rate": rate("source_leak_text"),
        "source_leak_proto_rate": rate("source_leak_proto"),
        "mean_text_target_score": mean("text_target_score"),
        "mean_text_target_minus_source": mean("text_target_minus_source"),
        "mean_proto_target_score": mean("proto_target_score"),
        "mean_proto_target_minus_source": mean("proto_target_minus_source"),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_bar_plot(summary_by_method, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("text_top1_target_rate", "Text top-1 target"),
        ("proto_top1_target_rate", "Proto top-1 target"),
        ("source_leak_text_rate", "Text source leak"),
        ("source_leak_proto_rate", "Proto source leak"),
    ]
    methods = list(summary_by_method)
    x = np.arange(len(methods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2), 5))
    for i, (key, label) in enumerate(metrics):
        vals = [
            summary_by_method[m].get(key)
            if summary_by_method[m].get(key) is not None else 0.0
            for m in methods
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Generated Image Semantic Diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_margin_histogram(rows_by_method, output_path, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    any_values = False
    for method, rows in rows_by_method.items():
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        any_values = True
        ax.hist(vals, bins=50, alpha=0.45, label=method, density=True)
    if not any_values:
        plt.close(fig)
        return
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(key.replace("_", " "))
    ax.set_xlabel("target score - source score")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_grid(rows, output_path, title, score_key, max_images):
    selected = rows[:max_images]
    if not selected:
        return
    thumb_w, thumb_h, caption_h = 128, 128, 62
    cols = min(6, len(selected))
    rows_n = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + caption_h)),
                       "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(selected):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + caption_h)
        try:
            img = Image.open(row["path"]).convert("RGB").resize((thumb_w, thumb_h))
        except Exception:
            continue
        canvas.paste(img, (x, y))
        score = row.get(score_key)
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        caption = [
            f"T:{row['target_name'][:14]}",
            f"S:{row['source_name'][:14]}" if row.get("source_name") else "",
            f"{score_key[:8]}={score_text}",
        ]
        draw.text((x + 3, y + thumb_h + 3), "\n".join([c for c in caption if c]),
                  fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 0, 0))
    canvas.save(output_path)


def source_leakage_sort_key(row):
    """Sort rows so the strongest source-over-target leakage comes first.

    Methods like BarePrompt do not have a source class, so their source-margin
    fields are None. Put those rows at the end instead of asking Python to
    compare None values.
    """
    proto_margin = row.get("proto_target_minus_source")
    if isinstance(proto_margin, (int, float)):
        return proto_margin
    text_margin = row.get("text_target_minus_source")
    if isinstance(text_margin, (int, float)):
        return text_margin
    return float("inf")


def per_class_summary(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["method"], r["label"], r["target_name"])].append(r)

    out = []
    for (method, label, target_name), group in sorted(grouped.items()):
        summary = summarize_rows(group)
        out.append({
            "method": method,
            "label": label,
            "target_name": target_name,
            **summary,
        })
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method_names and len(args.method_names) != len(args.generated_dirs):
        raise ValueError("--method_names must have the same length as --generated_dirs")
    method_names = args.method_names or [
        Path(p).name[:40] for p in args.generated_dirs
    ]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU")
        device = "cpu"

    print("=" * 70)
    print("Generated Image Semantic Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {method_names}")
    print(f"Output: {output_dir}")

    from augment.head2tail_prompts import get_class_names
    class_names = get_class_names(args.dataset)
    num_classes = get_num_classes(args.dataset)
    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    clip_module, clip_model, preprocess = load_clip(args.clip_model, device)
    text_feats, text_prompts = encode_text_prompts(
        class_names, clip_module, clip_model, device, args.batch_size
    )

    proto_feats = None
    proto_present = None
    if not args.skip_real_prototypes:
        print("Computing real-image CLIP prototypes...")
        real_images, cls_num_list = load_real_class_images(args)
        proto_feats, proto_present = compute_real_prototypes(
            real_images, clip_model, preprocess, device, args.batch_size, num_classes
        )
        with open(output_dir / "real_class_counts.json", "w") as f:
            json.dump({str(i): int(n) for i, n in enumerate(cls_num_list)}, f, indent=2)

    all_rows = []
    rows_by_method = {}
    summary_by_method = {}

    for method, generated_dir in zip(method_names, args.generated_dirs):
        entries = flatten_metadata(generated_dir)
        n_total = len(entries)
        entries = sample_entries(entries, args.max_gen_per_method, args.seed)
        print(f"\n[{method}] {generated_dir}")
        print(f"  Scoring {len(entries)}/{n_total} generated images")

        rows = score_entries(
            entries, method, class_names, text_feats, proto_feats, proto_present,
            clip_model, preprocess, device, args.batch_size
        )
        rows_by_method[method] = rows
        all_rows.extend(rows)
        summary = summarize_rows(rows)
        summary_by_method[method] = summary
        print(json.dumps(summary, indent=2))

        write_csv(output_dir / f"{method}_sample_scores.csv", rows)

        if any(
            isinstance(r.get("proto_target_minus_source"), (int, float))
            or isinstance(r.get("text_target_minus_source"), (int, float))
            for r in rows
        ):
            leak_sorted = sorted(rows, key=source_leakage_sort_key)
            make_grid(
                leak_sorted,
                output_dir / f"{method}_worst_source_leakage.jpg",
                f"{method}: worst source leakage",
                "proto_target_minus_source",
                args.max_grid_images,
            )

        rng = random.Random(args.seed)
        random_rows = list(rows)
        rng.shuffle(random_rows)
        make_grid(
            random_rows,
            output_dir / f"{method}_random_grid.jpg",
            f"{method}: random generated samples",
            "proto_target_score",
            args.max_grid_images,
        )

    write_csv(output_dir / "all_sample_scores.csv", all_rows)
    write_csv(output_dir / "per_class_summary.csv", per_class_summary(all_rows))
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_by_method, f, indent=2)

    make_bar_plot(summary_by_method, output_dir / "method_comparison.png")
    make_margin_histogram(
        rows_by_method, output_dir / "text_target_minus_source_hist.png",
        "text_target_minus_source"
    )
    make_margin_histogram(
        rows_by_method, output_dir / "proto_target_minus_source_hist.png",
        "proto_target_minus_source"
    )

    print("\nSaved diagnostics:")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'all_sample_scores.csv'}")
    print(f"  {output_dir / 'per_class_summary.csv'}")
    print(f"  {output_dir / 'method_comparison.png'}")
    print(f"  {output_dir / '*_worst_source_leakage.jpg'}")


if __name__ == "__main__":
    main()
