"""Empirical fisheye correction profiles for PavAnalytics.

These profiles are intended for reproducible preprocessing of known camera/frame
configurations when a checkerboard calibration file is not being used.
They are not substitutes for measured checkerboard calibration parameters.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


PAVANALYTICS_2048X1152_K = np.array(
    [
        [560.0, 0.0, 960.0],
        [0.0, 560.0, 576.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

PAVANALYTICS_2048X1152_D = np.array(
    [[0.0], [0.0], [0.0], [0.0]],
    dtype=np.float64,
)

PAVANALYTICS_INPUT_SIZE = (2048, 1152)
PAVANALYTICS_OUTPUT_SIZE = (720, 720)
PAVANALYTICS_OUTPUT_FOV_DEG = 75.0


def build_pavanalytics_profile_maps(
    output_size: tuple[int, int] = PAVANALYTICS_OUTPUT_SIZE,
    output_fov_deg: float = PAVANALYTICS_OUTPUT_FOV_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build rectification maps for the empirically tuned PavAnalytics profile.

    The output camera is explicitly rectilinear. The 75-degree field of view was
    tuned on the 2048x1152 fisheye frame used during local validation.
    """
    out_w, out_h = (int(output_size[0]), int(output_size[1]))
    if out_w <= 0 or out_h <= 0:
        raise ValueError("output_size values must be positive")
    if not 1.0 < float(output_fov_deg) < 179.0:
        raise ValueError("output_fov_deg must be between 1 and 179 degrees")

    focal = (out_w / 2.0) / math.tan(math.radians(float(output_fov_deg)) / 2.0)
    new_K = np.array(
        [
            [focal, 0.0, (out_w - 1) / 2.0],
            [0.0, focal, (out_h - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    identity = np.eye(3, dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        PAVANALYTICS_2048X1152_K,
        PAVANALYTICS_2048X1152_D,
        identity,
        new_K,
        (out_w, out_h),
        cv2.CV_32FC1,
    )
    return map1, map2, new_K


def correct_with_pavanalytics_profile(
    frame: np.ndarray,
    output_size: tuple[int, int] = PAVANALYTICS_OUTPUT_SIZE,
    output_fov_deg: float = PAVANALYTICS_OUTPUT_FOV_DEG,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Rectify a 2048x1152 fisheye frame using the tuned local profile."""
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    frame_size = (frame.shape[1], frame.shape[0])
    if frame_size != PAVANALYTICS_INPUT_SIZE:
        raise ValueError(
            "The tuned PavAnalytics profile expects a 2048x1152 input frame; "
            f"received {frame_size}."
        )

    map1, map2, _ = build_pavanalytics_profile_maps(
        output_size=output_size,
        output_fov_deg=output_fov_deg,
    )
    return cv2.remap(
        frame,
        map1,
        map2,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
