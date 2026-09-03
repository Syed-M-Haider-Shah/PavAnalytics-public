"""Fisheye camera calibration utilities for PavAnalytics.

The implementation follows the OpenCV fisheye calibration workflow used in the
PavAnalytics methodology: detect checkerboard corners, estimate the camera
intrinsic matrix K and fisheye distortion coefficients D, and save the
calibration for later remapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


DEFAULT_CHECKERBOARD_SIZE = (8, 6)  # OpenCV convention: inner corners (columns, rows)
DEFAULT_SQUARE_SIZE = 1.0
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class CalibrationResult:
    rms: float
    K: np.ndarray
    D: np.ndarray
    image_size: tuple[int, int]
    checkerboard_size: tuple[int, int]
    square_size: float
    images_used: int


def collect_image_paths(path: str | Path) -> list[Path]:
    """Collect calibration images from one file or a directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Calibration image path does not exist: {path}")

    images = [
        p for p in sorted(path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No calibration images found in: {path}")
    return images


def _object_points(checkerboard_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = checkerboard_size
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def detect_checkerboard_corners(
    image: np.ndarray,
    checkerboard_size: tuple[int, int] = DEFAULT_CHECKERBOARD_SIZE,
) -> np.ndarray | None:
    """Detect and refine checkerboard inner corners in one image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.astype(np.float64).reshape(1, -1, 2)


def calibrate_fisheye(
    image_paths: Sequence[str | Path] | Iterable[str | Path],
    checkerboard_size: tuple[int, int] = DEFAULT_CHECKERBOARD_SIZE,
    square_size: float = DEFAULT_SQUARE_SIZE,
) -> CalibrationResult:
    """Estimate fisheye intrinsic matrix K and distortion coefficients D.

    Parameters
    ----------
    image_paths:
        Calibration images containing the same checkerboard.
    checkerboard_size:
        Number of *inner corners* as ``(columns, rows)``. The project method
        describes an 8 x 6 checkerboard, so ``(8, 6)`` is the default.
    square_size:
        Physical checker square size. Any consistent unit can be used.
    """
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("At least one calibration image is required.")
    if square_size <= 0:
        raise ValueError("square_size must be greater than zero.")

    objp = _object_points(checkerboard_size, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                "All calibration images must have the same resolution. "
                f"Expected {image_size}, got {current_size} for {image_path}."
            )

        corners = detect_checkerboard_corners(image, checkerboard_size)
        if corners is None:
            continue

        object_points.append(objp.copy())
        image_points.append(corners)

    if image_size is None:
        raise ValueError("None of the calibration images could be read.")
    if len(image_points) < 3:
        raise ValueError(
            "Fisheye calibration needs checkerboard detections from multiple views. "
            f"Only {len(image_points)} valid view(s) were detected."
        )

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in image_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in image_points]

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    rms, K, D, _, _ = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        flags,
        criteria,
    )

    return CalibrationResult(
        rms=float(rms),
        K=K,
        D=D,
        image_size=image_size,
        checkerboard_size=tuple(checkerboard_size),
        square_size=float(square_size),
        images_used=len(image_points),
    )


def save_calibration(result: CalibrationResult, output_path: str | Path) -> Path:
    """Save calibration parameters to a compressed NumPy archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        K=result.K,
        D=result.D,
        image_size=np.asarray(result.image_size, dtype=np.int32),
        checkerboard_size=np.asarray(result.checkerboard_size, dtype=np.int32),
        square_size=np.asarray(result.square_size, dtype=np.float64),
        rms=np.asarray(result.rms, dtype=np.float64),
        images_used=np.asarray(result.images_used, dtype=np.int32),
    )
    return output_path


def load_calibration(calibration_path: str | Path) -> CalibrationResult:
    """Load calibration parameters saved by :func:`save_calibration`."""
    calibration_path = Path(calibration_path)
    with np.load(calibration_path) as data:
        return CalibrationResult(
            rms=float(data["rms"]),
            K=np.asarray(data["K"], dtype=np.float64),
            D=np.asarray(data["D"], dtype=np.float64),
            image_size=tuple(int(v) for v in data["image_size"]),
            checkerboard_size=tuple(int(v) for v in data["checkerboard_size"]),
            square_size=float(data["square_size"]),
            images_used=int(data["images_used"]),
        )
