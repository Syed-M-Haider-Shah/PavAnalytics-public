"""Command-line entry point for PavAnalytics fisheye calibration and correction."""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from .calibration import DEFAULT_CHECKERBOARD_SIZE, calibrate_fisheye, collect_image_paths, load_calibration, save_calibration
from .correction import build_rectification_maps
from .profiles import PAVANALYTICS_OUTPUT_FOV_DEG, PAVANALYTICS_OUTPUT_SIZE, correct_with_pavanalytics_profile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _calibrate(args):
    images = collect_image_paths(args.images)
    result = calibrate_fisheye(images, checkerboard_size=(args.cols, args.rows), square_size=args.square_size)
    save_calibration(result, args.output)
    print(f"Calibration saved to: {args.output}")
    print(f"Valid checkerboard views: {result.images_used}")
    print(f"RMS reprojection error: {result.rms:.6f}")
    print("K="); print(result.K); print("D="); print(result.D.ravel())

def _correct(args):
    calibration = load_calibration(args.calibration)
    input_path = Path(args.input); output_path = Path(args.output)
    if input_path.is_file():
        files = [input_path]; output_path.parent.mkdir(parents=True, exist_ok=True); single_file_output = output_path
    elif input_path.is_dir():
        files = [p for p in sorted(input_path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files: raise FileNotFoundError(f"No images found in: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True); single_file_output = None
    else: raise FileNotFoundError(f"Input path does not exist: {input_path}")
    map1, map2, _ = build_rectification_maps(calibration.K, calibration.D, calibration.image_size, balance=args.balance)
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: continue
        frame_size = (image.shape[1], image.shape[0])
        if frame_size != calibration.image_size: raise ValueError(f"Image {path} has resolution {frame_size}, expected {calibration.image_size}.")
        corrected = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        destination = single_file_output if single_file_output else output_path / path.name
        if not cv2.imwrite(str(destination), corrected): raise OSError(f"Failed to write: {destination}")
        print(f"Saved: {destination}")

def _correct_profile(args):
    input_path = Path(args.input); output_path = Path(args.output)
    if input_path.is_file(): files=[input_path]; output_path.parent.mkdir(parents=True, exist_ok=True); single_file_output=output_path
    elif input_path.is_dir():
        files=[p for p in sorted(input_path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files: raise FileNotFoundError(f"No images found in: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True); single_file_output=None
    else: raise FileNotFoundError(f"Input path does not exist: {input_path}")
    for path in files:
        image=cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: continue
        corrected=correct_with_pavanalytics_profile(image, output_size=(args.width,args.height), output_fov_deg=args.fov)
        destination=single_file_output if single_file_output else output_path/path.name
        if not cv2.imwrite(str(destination), corrected): raise OSError(f"Failed to write: {destination}")
        print(f"Saved: {destination}")

def build_parser():
    parser=argparse.ArgumentParser(description="PavAnalytics fisheye calibration and correction")
    subparsers=parser.add_subparsers(dest="command", required=True)
    p=subparsers.add_parser("calibrate"); p.add_argument("--images",required=True); p.add_argument("--output",default="fisheye_calibration.npz"); p.add_argument("--cols",type=int,default=DEFAULT_CHECKERBOARD_SIZE[0]); p.add_argument("--rows",type=int,default=DEFAULT_CHECKERBOARD_SIZE[1]); p.add_argument("--square-size",type=float,default=1.0); p.set_defaults(func=_calibrate)
    p=subparsers.add_parser("correct"); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--calibration",required=True); p.add_argument("--balance",type=float,default=0.0); p.set_defaults(func=_correct)
    p=subparsers.add_parser("correct-profile"); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--width",type=int,default=PAVANALYTICS_OUTPUT_SIZE[0]); p.add_argument("--height",type=int,default=PAVANALYTICS_OUTPUT_SIZE[1]); p.add_argument("--fov",type=float,default=PAVANALYTICS_OUTPUT_FOV_DEG); p.set_defaults(func=_correct_profile)
    return parser

def main():
    args=build_parser().parse_args(); args.func(args)

if __name__ == "__main__": main()
