from pathlib import Path

import torch
from transformers import AutoModelForImageClassification


def _latest_epoch_key(checkpoint):
    epoch_keys = [key for key in checkpoint if key.startswith("epoch_")]
    if not epoch_keys:
        raise KeyError("No epoch checkpoints were found in the checkpoint file.")
    return max(epoch_keys, key=lambda key: int(key.split("_")[-1]))


def load_model(config, device, checkpoint_file=None):
    # Load the pre-trained Swin Transformer model from Hugging Face
    model_name = config.model_name
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=5,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # If a repository-local checkpoint already exists, reuse it as the
    # source checkpoint in the same way as the original script: load all
    # non-classifier weights and leave the classifier freshly initialised.
    if checkpoint_file is not None and Path(checkpoint_file).is_file():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch_key = _latest_epoch_key(checkpoint)
        print(f"Reusing existing checkpoint: {checkpoint_file} ({epoch_key})")

        # Filter out classifier layer weights from the checkpoint
        filtered_checkpoint = {
            k: v for k, v in checkpoint[epoch_key].items()
            if not k.startswith("classifier")
        }

        # Load weights except for the classifier
        model.load_state_dict(filtered_checkpoint, strict=False)

        # Check if weights are properly loaded by comparing a few random layers
        mismatches = 0
        for name, param in model.named_parameters():
            if name in filtered_checkpoint:
                if not torch.equal(param.data, filtered_checkpoint[name]):
                    print(f"Mismatch found in layer: {name}")
                    mismatches += 1

        if mismatches == 0:
            print("All non-classifier layers are loaded correctly from the checkpoint.")
        else:
            print(f"{mismatches} layers have mismatched weights compared to the checkpoint.")
    else:
        print("No existing repository checkpoint found. Starting from the Hugging Face pretrained model.")

    # Unfreeze the last two layers for fine-tuning
    for param in model.swin.encoder.layers[:-2].parameters():
        param.requires_grad = False

    return model


def load_checkpoint_for_evaluation(model, checkpoint_file, device, epoch_key=None):
    """Load a saved training checkpoint into the model for evaluation."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if epoch_key is None:
        epoch_key = _latest_epoch_key(checkpoint)

    if epoch_key not in checkpoint:
        raise KeyError(
            f"Checkpoint '{epoch_key}' was not found in {checkpoint_file}. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(checkpoint[epoch_key], strict=True)
    model.to(device)
    print(f"Loaded {epoch_key} from {checkpoint_file} for evaluation.")
    return model
