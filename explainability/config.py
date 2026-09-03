from pathlib import Path

import torch

EXPLAINABILITY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPLAINABILITY_DIR.parent
CLASSIFICATION_DIR = REPOSITORY_ROOT / "classification"

MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
NUM_LABELS = 5
CLASS_NAMES = ["1", "2", "3", "4", "5"]
IMAGE_SIZE = 224

CLASSIFICATION_CHECKPOINT = (
    CLASSIFICATION_DIR
    / "checkpoints"
    / "all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth"
)
OUTPUT_DIR = EXPLAINABILITY_DIR / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
