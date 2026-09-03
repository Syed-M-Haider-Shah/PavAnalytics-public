"""PavAnalytics classification experiment runner.

Runs the existing Swin training/validation/test pipeline and can generate
Grad-CAM outputs for every image in the test set. The model and training logic
remain in the existing ``classification`` and ``explainability`` modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_scheduler

from classification.config import config, device
from classification.dataset import CustomImageDataset
from classification.evaluate import test_model
from classification.model import load_checkpoint_for_evaluation, load_model
from classification.train import train_model
from classification.transforms import data_transforms_train, data_transforms_val_test
from explainability.gradcam import GradCAM, prepare_image, save_gradcam_outputs
from explainability.model_loader import get_swin_gradcam_target_layer


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "classification"
    / "checkpoints"
    / "all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth"
)
DEFAULT_RESULTS = Path(__file__).resolve().parent / "classification" / "training_results_2layer_2e-5_retrain-with-spili-1.csv"
DEFAULT_GRADCAM_DIR = Path(__file__).resolve().parent / "outputs" / "gradcam_test"
CLASS_NAMES = ["1", "2", "3", "4", "5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test the PavAnalytics Swin classifier and generate test-set Grad-CAMs."
    )
    parser.add_argument("--mode", choices=["all", "train", "test", "gradcam"], default="all")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epoch", default=None, help="Checkpoint key such as epoch_50. Defaults to latest saved epoch.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--gradcam-output", type=Path, default=DEFAULT_GRADCAM_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def require_dir(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"--{label} is required for this mode.")
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")
    return path


def make_dataset(path: Path, training: bool) -> CustomImageDataset:
    transform = data_transforms_train if training else data_transforms_val_test
    dataset = CustomImageDataset(root_dir=str(path), transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No valid images were found under {path}. Expected class folders 1-5.")
    return dataset


def make_loader(dataset: CustomImageDataset, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def load_full_checkpoint(model, checkpoint: Path, epoch: str | None):
    return load_checkpoint_for_evaluation(
        model,
        checkpoint,
        device,
        epoch_key=epoch,
    )


def run_gradcam_for_test_set(model, test_dataset: CustomImageDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    target_layer = get_swin_gradcam_target_layer(model)

    with GradCAM(model, target_layer) as gradcam:
        for image_path, true_index in zip(test_dataset.image_paths, test_dataset.labels):
            image_path = Path(image_path)
            display_image, input_tensor = prepare_image(image_path, image_size=224)
            input_tensor = input_tensor.to(device)
            result = gradcam.generate(input_tensor)

            predicted_index = result["predicted_index"]
            true_label = CLASS_NAMES[int(true_index)]
            predicted_label = CLASS_NAMES[predicted_index]
            correctness = "correct" if int(true_index) == predicted_index else "incorrect"

            image_output = output_dir / correctness / f"true_{true_label}_pred_{predicted_label}"
            paths = save_gradcam_outputs(
                display_image,
                result["cam"],
                image_output,
                image_path.stem,
            )
            print(
                f"Grad-CAM: {image_path.name} | true={true_label} | "
                f"predicted={predicted_label} | confidence={result['confidence']:.4f} | "
                f"overlay={paths['overlay']}"
            )


def main() -> None:
    args = parse_args()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = val_dataset = test_dataset = None
    model = None

    if args.mode in {"all", "train"}:
        if device.type != "cuda":
            raise RuntimeError(
                "The original PavAnalytics training loop uses CUDA autocast. "
                "Run training on a CUDA-capable environment."
            )

        train_dir = require_dir(args.train_dir, "train-dir")
        val_dir = require_dir(args.val_dir, "val-dir")
        train_dataset = make_dataset(train_dir, training=True)
        val_dataset = make_dataset(val_dir, training=False)
        train_loader = make_loader(train_dataset, shuffle=True, num_workers=args.num_workers)
        val_loader = make_loader(val_dataset, shuffle=False, num_workers=args.num_workers)

        print(f"Training dataset size: {len(train_dataset)} images")
        print(f"Validation dataset size: {len(val_dataset)} images")

        model = load_model(config, device, checkpoint_file=args.checkpoint)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * config.epochs,
        )
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        results: list[dict] = []

        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config.epochs,
            args.checkpoint,
            scaler,
            results,
        )

        args.results.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.results, index=False)
        print(f"Training results saved to: {args.results}")

        evaluation_epoch = args.epoch or f"epoch_{config.epochs}"
        model = load_full_checkpoint(model, args.checkpoint, evaluation_epoch)

    if args.mode in {"all", "test", "gradcam"}:
        test_dir = require_dir(args.test_dir, "test-dir")
        test_dataset = make_dataset(test_dir, training=False)
        print(f"Testing dataset size: {len(test_dataset)} images")

        if model is None:
            model = load_model(config, device, checkpoint_file=None)
            model = load_full_checkpoint(model, args.checkpoint, args.epoch)

    if args.mode in {"all", "test"}:
        test_loader = make_loader(test_dataset, shuffle=False, num_workers=args.num_workers)
        test_model(model, test_loader)

    if args.mode in {"all", "gradcam"}:
        run_gradcam_for_test_set(model, test_dataset, args.gradcam_output)


if __name__ == "__main__":
    main()
