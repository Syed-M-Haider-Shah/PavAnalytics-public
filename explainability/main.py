import argparse
from pathlib import Path

from .config import (
    CLASSIFICATION_CHECKPOINT,
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    MODEL_NAME,
    NUM_LABELS,
    OUTPUT_DIR,
)
from .gradcam import GradCAM, prepare_image, save_gradcam_outputs
from .model_loader import get_swin_gradcam_target_layer, load_swin_classifier


_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for the PavAnalytics Swin classifier."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=Path, help="Single image to explain.")
    source_group.add_argument("--input-dir", type=Path, help="Folder of images to explain.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CLASSIFICATION_CHECKPOINT,
        help="Classification checkpoint file. Defaults to the repository-local checkpoint.",
    )
    parser.add_argument(
        "--epoch",
        default=None,
        help="Checkpoint key such as epoch_50. Defaults to the latest saved epoch.",
    )
    parser.add_argument(
        "--class-label",
        choices=CLASS_NAMES,
        default=None,
        help="Optional pavement class label (1-5). Defaults to the model prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for Grad-CAM outputs.",
    )
    return parser.parse_args()


def _collect_images(args):
    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")
        return [args.image]

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")

    images = [
        path
        for path in sorted(args.input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")
    return images


def main():
    args = _parse_args()
    images = _collect_images(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    model, epoch_key = load_swin_classifier(
        MODEL_NAME,
        NUM_LABELS,
        args.checkpoint,
        DEVICE,
        epoch_key=args.epoch,
    )
    target_layer = get_swin_gradcam_target_layer(model)

    target_index = None
    if args.class_label is not None:
        target_index = CLASS_NAMES.index(args.class_label)

    with GradCAM(model, target_layer) as gradcam:
        for image_path in images:
            display_image, input_tensor = prepare_image(image_path, IMAGE_SIZE)
            input_tensor = input_tensor.to(DEVICE)

            result = gradcam.generate(input_tensor, class_index=target_index)
            paths = save_gradcam_outputs(
                display_image,
                result["cam"],
                args.output_dir,
                image_path.stem,
            )

            predicted_label = CLASS_NAMES[result["predicted_index"]]
            target_label = CLASS_NAMES[result["target_index"]]
            print(
                f"{image_path.name}: predicted class={predicted_label}, "
                f"confidence={result['confidence']:.4f}, "
                f"Grad-CAM target={target_label}, checkpoint={epoch_key}"
            )
            print(f"  Overlay: {paths['overlay']}")


if __name__ == "__main__":
    main()
