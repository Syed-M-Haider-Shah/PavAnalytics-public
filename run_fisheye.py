"""Automatic PavAnalytics fisheye-folder preprocessing.

The automatic detector is deliberately conservative and targets the strong dark
circular/oval border produced by the PavAnalytics fisheye footage. Images that
do not meet the fisheye criteria are copied unchanged. Detected 2048x1152
fisheye frames are rectified with the empirically tuned PavAnalytics profile.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import cv2
import numpy as np

from fisheye_correction.profiles import (
    PAVANALYTICS_INPUT_SIZE,
    correct_with_pavanalytics_profile,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and correct fisheye images in a folder.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--fov", type=float, default=75.0)
    parser.add_argument("--force", choices=["auto", "always", "never"], default="auto")
    return parser.parse_args()


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [p for p in sorted(iterator) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")
    return images


def fisheye_signature(image: np.ndarray, dark_threshold: int = 25) -> dict[str, float | bool]:
    """Measure the dark-border signature visible in the PavAnalytics fisheye footage."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    border_width = max(2, int(min(height, width) * 0.04))
    corner_h = max(2, int(height * 0.12))
    corner_w = max(2, int(width * 0.12))

    border = np.concatenate(
        [
            gray[:border_width, :].ravel(),
            gray[-border_width:, :].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        ]
    )
    corners = np.concatenate(
        [
            gray[:corner_h, :corner_w].ravel(),
            gray[:corner_h, -corner_w:].ravel(),
            gray[-corner_h:, :corner_w].ravel(),
            gray[-corner_h:, -corner_w:].ravel(),
        ]
    )

    border_dark_fraction = float(np.mean(border < dark_threshold))
    corner_dark_fraction = float(np.mean(corners < dark_threshold))
    likely_fisheye = border_dark_fraction >= 0.25 and corner_dark_fraction >= 0.45

    return {
        "likely_fisheye": likely_fisheye,
        "border_dark_fraction": border_dark_fraction,
        "corner_dark_fraction": corner_dark_fraction,
    }


def main() -> None:
    args = parse_args()
    images = collect_images(args.input_dir, args.recursive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []

    for image_path in images:
        relative = image_path.relative_to(args.input_dir)
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        metrics = fisheye_signature(image)
        if args.force == "always":
            should_correct = True
        elif args.force == "never":
            should_correct = False
        else:
            should_correct = bool(metrics["likely_fisheye"])

        frame_size = (image.shape[1], image.shape[0])
        action = "copied_unchanged"

        if should_correct:
            if frame_size != PAVANALYTICS_INPUT_SIZE:
                print(
                    f"Detected likely fisheye image but did not apply the tuned profile because "
                    f"{image_path.name} has size {frame_size}; expected {PAVANALYTICS_INPUT_SIZE}."
                )
                shutil.copy2(image_path, destination)
                action = "fisheye_detected_unsupported_size"
            else:
                corrected = correct_with_pavanalytics_profile(image, output_fov_deg=args.fov)
                if not cv2.imwrite(str(destination), corrected):
                    raise OSError(f"Failed to write corrected image: {destination}")
                action = "fisheye_corrected"
        else:
            shutil.copy2(image_path, destination)

        print(
            f"{image_path.name}: {action} | border_dark={metrics['border_dark_fraction']:.3f} | "
            f"corner_dark={metrics['corner_dark_fraction']:.3f}"
        )
        log_rows.append(
            {
                "image": str(relative),
                "width": frame_size[0],
                "height": frame_size[1],
                "border_dark_fraction": metrics["border_dark_fraction"],
                "corner_dark_fraction": metrics["corner_dark_fraction"],
                "detected_fisheye": metrics["likely_fisheye"],
                "action": action,
            }
        )

    log_path = args.output_dir / "fisheye_processing_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Processing log saved to: {log_path}")


if __name__ == "__main__":
    main()
