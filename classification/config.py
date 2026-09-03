from pathlib import Path

import torch
import wandb

# Repository-relative paths
CLASSIFICATION_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = CLASSIFICATION_DIR / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth"
RESULTS_FILE = CLASSIFICATION_DIR / "training_results_2layer_2e-5_retrain-with-spili-1.csv"

# Initialize Weights & Biases
wandb.init(project='image-classification', config={
    "learning_rate": 2e-5,
    "epochs": 50,
    "batch_size": 64,
    "model_name": "microsoft/swin-tiny-patch4-window7-224"
})

config = wandb.config

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
