from pathlib import Path

import torch

SEGMENTATION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SEGMENTATION_DIR.parent
DATA_DIR = REPOSITORY_ROOT / "data"
SOURCE_IMAGES_DIR = DATA_DIR / "1"
OUTPUT_SEGMENTS_DIR = DATA_DIR / "segmented_images"
CHECKPOINT_DIR = SEGMENTATION_DIR / "checkpoints" / "final_model"
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
