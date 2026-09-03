"""Fisheye frame rectification for PavAnalytics.

Uses OpenCV's fisheye new-camera-matrix estimation,
``initUndistortRectifyMap`` and ``remap``. The ``balance`` parameter controls
field-of-view retention versus cropping, matching the method described for the
CRSI preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import CalibrationResult, load_calibration


def build_rectification_maps(
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
    balance: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build OpenCV fisheye rectification maps.

    ``balance=0`` favours cropping/less black border; ``balance=1`` retains more
    of the original fisheye field of view. Values between 0 and 1 interpolate
    between those behaviours.
    """
    if not 0.0 <= balance <= 1.0:
        raise ValueError("balance must be between 0.0 and 1.0.")

    image_size = tuple(int(v) for v in image_size)
    output_size = image_size if output_size is None else tuple(int(v) for v in output_size)

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(4, 1)
    identity = np.eye(3, dtype=np.float64)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        image_size,
        identity,
        balance=float(balance),
        new_size=output_size,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        identity,
        new_K,
        output_size,
        cv2.CV_16SC2,
    )
    return map1, map2, new_K


def undistort_frame(
    frame: np.ndarray,
    calibration: CalibrationResult,
    balance: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Undistort one fisheye frame using a saved camera calibration."""
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")

    frame_size = (frame.shape[1], frame.shape[0])
    if frame_size != calibration.image_size:
        raise ValueError(
            "Input frame resolution must match the calibration resolution. "
            f"Calibration: {calibration.image_size}; frame: {frame_size}."
        )

    map1, map2, _ = build_rectification_maps(
        calibration.K,
        calibration.D,
        calibration.image_size,
        balance=balance,
    )
    return cv2.remap(
        frame,
        map1,
        map2,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )


def undistort_image_file(
    input_path: str | Path,
    output_path: str | Path,
    calibration_path: str | Path,
    balance: float = 0.0,
) -> Path:
    """Undistort one image file and save the corrected result."""
    calibration = load_calibration(calibration_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    corrected = undistort_frame(image, calibration, balance=balance)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), corrected):
        raise OSError(f"Failed to write corrected image: {output_path}")
    return output_path
