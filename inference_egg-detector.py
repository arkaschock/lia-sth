#!/usr/bin/env python3
# pip install ultralytics sahi pillow numpy tqdm

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch tiled inference with Ultralytics YOLOv11 and SAHI."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder with input images (jpg/jpeg/png/tif/tiff).",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to Ultralytics YOLOv11 .pt weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string (cpu, mps, cuda, cuda:0, etc.). Defaults to auto.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Inference image size (imgsz).",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="Allowed extensions (e.g., jpg jpeg png tif tiff).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help=(
            "IoU threshold for SAHI slice postprocessing. "
            "When set, SAHI uses IOU matching with this threshold."
        ),
    )
    return parser.parse_args()


def normalize_exts(exts):
    if not exts:
        exts = ["jpg", "jpeg", "png", "tif", "tiff"]
    normalized = set()
    for ext in exts:
        for part in str(ext).split(","):
            part = part.strip().lower()
            if not part:
                continue
            if not part.startswith("."):
                part = f".{part}"
            normalized.add(part)
    return normalized


def select_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def color_for_class(class_name: str):
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    r, g, b = digest[0], digest[1], digest[2]
    min_v = 60
    r = int(r / 255 * (255 - min_v) + min_v)
    g = int(g / 255 * (255 - min_v) + min_v)
    b = int(b / 255 * (255 - min_v) + min_v)
    return (r, g, b)


def class_name_from_prediction(obj) -> str:
    class_name = "object"
    if getattr(obj, "category", None) is not None:
        class_name = getattr(obj.category, "name", None) or class_name
    return str(class_name)


def compute_embryonated_percent(em_count: int, total_count: int, infer_count: int):
    denominator = total_count - infer_count
    if denominator <= 0:
        return None, denominator
    return (em_count / denominator) * 100.0, denominator


def format_class_counts(class_counts: Counter) -> str:
    if not class_counts:
        return ""
    return ";".join(f"{name}:{class_counts[name]}" for name in sorted(class_counts))


def to_grayscale_rgb(rgb_np: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * rgb_np[..., 0]
        + 0.587 * rgb_np[..., 1]
        + 0.114 * rgb_np[..., 2]
    ).astype(np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    return gray_rgb


def draw_predictions(image: Image.Image, predictions) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    thickness = max(6, int(min(width, height) / 180))
    font_size = min(44, max(12, int(min(width, height) * 0.013)))
    text_pad = max(4, thickness // 2)
    font = load_font(font_size)

    for obj in predictions:
        try:
            x1 = int(max(0, obj.bbox.minx))
            y1 = int(max(0, obj.bbox.miny))
            x2 = int(min(width - 1, obj.bbox.maxx))
            y2 = int(min(height - 1, obj.bbox.maxy))
        except Exception:
            continue

        class_name = class_name_from_prediction(obj)
        score_val = None
        if getattr(obj, "score", None) is not None:
            score_val = getattr(obj.score, "value", None)

        label = class_name
        if score_val is not None:
            label = f"{class_name} {score_val:.2f}"

        color = color_for_class(class_name)

        for t in range(thickness):
            draw.rectangle(
                [x1 - t, y1 - t, x2 + t, y2 + t],
                outline=color,
            )

        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(label, font=font)

        text_x = x1
        text_y = max(0, y1 - text_h - text_pad)
        if text_x + text_w + (2 * text_pad) > width:
            text_x = max(0, width - text_w - (2 * text_pad))
        if text_y + text_h + (2 * text_pad) > height:
            text_y = max(0, height - text_h - (2 * text_pad))

        draw.rectangle(
            [
                text_x,
                text_y,
                text_x + text_w + (2 * text_pad),
                text_y + text_h + (2 * text_pad),
            ],
            fill=color,
        )
        draw.text((text_x + text_pad, text_y + text_pad), label, fill=(255, 255, 255), font=font)

    return annotated


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 1

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model_path does not exist: {model_path}", file=sys.stderr)
        return 1

    exts = normalize_exts(args.ext)
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )

    output_dir = input_dir / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    detection_model = UltralyticsDetectionModel(
        model_path=str(model_path),
        confidence_threshold=args.conf,
        device=device,
        image_size=args.imgsz,
    )

    try:
        detection_model.image_size = args.imgsz
    except Exception:
        pass

    try:
        if hasattr(detection_model, "model") and hasattr(detection_model.model, "overrides"):
            detection_model.model.overrides["imgsz"] = args.imgsz
    except Exception:
        pass

    csv_rows = []
    total_detected = 0
    total_em = 0
    total_infer = 0

    for image_path in tqdm(image_paths, desc="Processing", unit="img"):
        try:
            with Image.open(image_path) as im:
                rgb_image = im.convert("RGB")
        except Exception as exc:
            print(f"Warning: failed to read {image_path.name}: {exc}", file=sys.stderr)
            continue

        rgb_np = np.array(rgb_image)
        gray_rgb = to_grayscale_rgb(rgb_np)

        try:
            sliced_kwargs = dict(
                slice_height=800,
                slice_width=800,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )
            if args.iou is not None:
                sliced_kwargs["postprocess_match_metric"] = "IOU"
                sliced_kwargs["postprocess_match_threshold"] = args.iou
            prediction = get_sliced_prediction(gray_rgb, detection_model, **sliced_kwargs)
        except Exception as exc:
            print(
                f"Warning: failed inference on {image_path.name}: {exc}",
                file=sys.stderr,
            )
            continue

        predictions = prediction.object_prediction_list or []
        annotated = draw_predictions(rgb_image, predictions)

        class_labels = [class_name_from_prediction(obj) for obj in predictions]
        class_counts = Counter(class_labels)
        em_count = sum(count for name, count in class_counts.items() if name.lower() == "em")
        infer_count = sum(count for name, count in class_counts.items() if name.lower() == "infer")
        embryo_pct, denominator = compute_embryonated_percent(
            em_count=em_count,
            total_count=len(class_labels),
            infer_count=infer_count,
        )

        output_path = output_dir / f"{image_path.stem}_det{image_path.suffix}"
        try:
            annotated.save(output_path)
        except Exception as exc:
            print(
                f"Warning: failed to save {output_path.name}: {exc}",
                file=sys.stderr,
            )
            continue

        csv_rows.append(
            {
                "image_name": image_path.name,
                "output_image": str(output_path),
                "detected_classes": "|".join(class_labels),
                "total_detected": len(class_labels),
                "class_counts": format_class_counts(class_counts),
                "em_count": em_count,
                "infer_count": infer_count,
                "all_detected_minus_infer": denominator,
                "embryonated_egg_percent": (
                    f"{embryo_pct:.6f}" if embryo_pct is not None else ""
                ),
            }
        )
        total_detected += len(class_labels)
        total_em += em_count
        total_infer += infer_count

        print(f"{image_path.name}\t{len(predictions)}\t{output_path}")

    total_pct, total_denominator = compute_embryonated_percent(
        em_count=total_em,
        total_count=total_detected,
        infer_count=total_infer,
    )
    csv_path = output_dir / "detection_labels.csv"
    fieldnames = [
        "image_name",
        "output_image",
        "detected_classes",
        "total_detected",
        "class_counts",
        "em_count",
        "infer_count",
        "all_detected_minus_infer",
        "embryonated_egg_percent",
    ]
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            writer.writerow(
                {
                    "image_name": "__TOTAL__",
                    "output_image": "",
                    "detected_classes": "",
                    "total_detected": total_detected,
                    "class_counts": "",
                    "em_count": total_em,
                    "infer_count": total_infer,
                    "all_detected_minus_infer": total_denominator,
                    "embryonated_egg_percent": (
                        f"{total_pct:.6f}" if total_pct is not None else ""
                    ),
                }
            )
    except Exception as exc:
        print(f"Warning: failed to write CSV {csv_path.name}: {exc}", file=sys.stderr)
    else:
        print(f"CSV\t{csv_path}")
        if total_pct is not None:
            print(f"TOTAL_EMBRYONATED_EGG_PERCENT\t{total_pct:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
