"""Batch road segmentation for PavAnalytics.

Given a folder, applies the trained two-class SegFormer model to every supported
image and saves the road-only image, binary road mask and a transparent overlay.
Class 1 is treated as road, matching the original PavAnalytics segmentation code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "segmentation" / "checkpoints" / "final_model"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROAD_CLASS = 1
IMG_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment road pixels in a folder of PavAnalytics images.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--recursive", action="store_true", help="Also process images in subfolders.")
    return parser.parse_args()


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")
    return images


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    road_colour = np.zeros_like(overlay)
    road_colour[..., 1] = 255.0
    selected = mask == ROAD_CLASS
    overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * road_colour[selected]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    images = collect_images(args.input_dir, args.recursive)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading segmentation checkpoint: {args.checkpoint}")

    processor = SegformerImageProcessor.from_pretrained(str(args.checkpoint))
    model = SegformerForSemanticSegmentation.from_pretrained(str(args.checkpoint))
    model.to(device)
    model.eval()

    for image_path in tqdm(images, desc="Segmenting images"):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = processor(image, size=IMG_SIZE, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            raw_logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(
                raw_logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        rgb = np.asarray(image, dtype=np.uint8)
        road_only = rgb * (pred_mask[..., None] == ROAD_CLASS)
        binary_mask = np.where(pred_mask == ROAD_CLASS, 255, 0).astype(np.uint8)
        overlay = overlay_mask(rgb, pred_mask)

        relative = image_path.relative_to(args.input_dir)
        destination_dir = args.output_dir / relative.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        Image.fromarray(road_only.astype(np.uint8)).save(destination_dir / f"{stem}_segment.png")
        Image.fromarray(binary_mask).save(destination_dir / f"{stem}_mask.png")
        Image.fromarray(overlay).save(destination_dir / f"{stem}_overlay.png")

    print(f"Done. Segmentation outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
