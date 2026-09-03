# Fisheye correction

This module implements the PavAnalytics fisheye preprocessing workflow:

1. Detect an 8 × 6 checkerboard in calibration images.
2. Estimate the fisheye intrinsic matrix `K` and distortion coefficients `D` with OpenCV.
3. Build rectification maps with `cv2.fisheye.initUndistortRectifyMap`.
4. Correct frames with `cv2.remap` before CRSI classification.
5. Use `balance` in the range 0 to 1 to control cropping versus field-of-view retention.

## Important checkerboard convention

OpenCV expects the checkerboard size as the number of **inner corners**, not the number of squares. The default is `(8, 6)` inner corners. If the physical board has 8 × 6 squares, use `--cols 7 --rows 5` instead.

## Calibrate

From the repository root:

```bash
python -m fisheye_correction.run calibrate \
  --images path/to/calibration_images \
  --output fisheye_calibration.npz \
  --cols 8 \
  --rows 6 \
  --square-size 1.0
```

Use the real physical checker-square size for `--square-size` if calibrated dimensions are required. For image undistortion alone, any consistent positive unit is sufficient.

## Correct one image

```bash
python -m fisheye_correction.run correct \
  --input path/to/fisheye_frame.jpg \
  --output path/to/corrected_frame.jpg \
  --calibration fisheye_calibration.npz \
  --balance 0.5
```

## Correct a directory

```bash
python -m fisheye_correction.run correct \
  --input path/to/fisheye_frames \
  --output path/to/corrected_frames \
  --calibration fisheye_calibration.npz \
  --balance 0.5
```

`balance=0` favours cropping and fewer invalid borders. `balance=1` retains more of the original fisheye field of view.

## Empirically tuned PavAnalytics profile

For the 2048×1152 fisheye frame used during local validation, an optional rectilinear profile is provided in `profiles.py`. It produces a 720×720 output with a 75° field of view:

```bash
python -m fisheye_correction.run correct-profile \
  --input path/to/fisheye_frame.jpg \
  --output path/to/corrected_frame.jpg
```

The profile uses:

```text
K = [[560,   0, 960],
     [  0, 560, 576],
     [  0,   0,   1]]
D = [0, 0, 0, 0]
output = 720x720
rectilinear FOV = 75 degrees
```

These are empirically tuned preprocessing values for the tested camera/frame configuration. They must not be reported as checkerboard-derived calibration parameters. For measured calibration, use the `calibrate` and `correct` commands instead.
