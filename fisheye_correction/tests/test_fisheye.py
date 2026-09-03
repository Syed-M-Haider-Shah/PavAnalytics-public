from pathlib import Path
import tempfile
import cv2
import numpy as np
from fisheye_correction.calibration import CalibrationResult, load_calibration, save_calibration
from fisheye_correction.correction import build_rectification_maps, undistort_frame

def test_rectification_maps_and_remap():
    width, height = 320, 240
    K = np.array([[220.0, 0.0, width / 2], [0.0, 220.0, height / 2], [0.0, 0.0, 1.0]])
    D = np.array([[-0.08], [0.01], [0.0], [0.0]], dtype=np.float64)
    calibration = CalibrationResult(rms=0.1, K=K, D=D, image_size=(width, height), checkerboard_size=(8, 6), square_size=1.0, images_used=10)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (280, 200), (255, 255, 255), 3)
    cv2.line(image, (0, height // 2), (width - 1, height // 2), (255, 255, 255), 2)
    map1, map2, new_K = build_rectification_maps(K, D, (width, height), balance=0.5)
    assert map1.shape[:2] == (height, width)
    assert map2.shape[:2] == (height, width)
    assert new_K.shape == (3, 3)
    corrected = undistort_frame(image, calibration, balance=0.5)
    assert corrected.shape == image.shape
    assert corrected.dtype == image.dtype

def test_calibration_round_trip():
    width, height = 320, 240
    result = CalibrationResult(rms=0.123, K=np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]), D=np.array([[-0.1], [0.01], [0.001], [0.0]]), image_size=(width, height), checkerboard_size=(8, 6), square_size=25.0, images_used=12)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration.npz"
        save_calibration(result, path)
        loaded = load_calibration(path)
        assert np.allclose(loaded.K, result.K)
        assert np.allclose(loaded.D, result.D)
        assert loaded.image_size == result.image_size
        assert loaded.checkerboard_size == result.checkerboard_size
        assert loaded.square_size == result.square_size
        assert loaded.images_used == result.images_used
