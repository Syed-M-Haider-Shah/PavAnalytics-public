# This model results are to be put in confrence paper.
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

from .config import config, device, CHECKPOINT_DIR, CHECKPOINT_FILE, RESULTS_FILE
from .dataset import CustomImageDataset
from .transforms import data_transforms_train, data_transforms_val_test
from .model import load_model, load_checkpoint_for_evaluation
from .train import train_model
from .evaluate import test_model


# Load the datasets
train_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Train", transform=data_transforms_train)
val_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Validation", transform=data_transforms_val_test)
test_dataset = CustomImageDataset(root_dir=r"D:\distributed_data_70_15_15-new\Test", transform=data_transforms_val_test)

# Print the count of images in each dataset
print(f"Training dataset size: {len(train_dataset)} images")
print(f"Validation dataset size: {len(val_dataset)} images")
print(f"Testing dataset size: {len(test_dataset)} images")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Create the repository-local checkpoint directory automatically
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Load the pre-trained Swin Transformer model. If this repository-local
# checkpoint already exists from an earlier run, reuse its non-classifier
# weights in the same way as the original experiment.
model = load_model(config, device, checkpoint_file=CHECKPOINT_FILE)

# Hugging Face default optimizer AdamW
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Hugging Face-style scheduler: linear decay without warmup steps
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * config.epochs
)

# CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

# Set up for mixed precision training
scaler = GradScaler()

# Prepare to store results in a list for saving as CSV
results = []

# Run training and save checkpoints inside classification/checkpoints/
train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    config.epochs,
    CHECKPOINT_FILE,
    scaler,
    results,
)

# Save results to CSV beside the classification code
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)
print(f"Training results saved to {RESULTS_FILE}")

# Reload the exact checkpoint saved by this run before test evaluation.
# This ensures evaluation uses the saved repository checkpoint rather than
# relying only on the model currently held in memory.
model = load_checkpoint_for_evaluation(
    model,
    CHECKPOINT_FILE,
    device,
    epoch_key=f"epoch_{config.epochs}",
)

# Test the model
test_model(model, test_loader)
