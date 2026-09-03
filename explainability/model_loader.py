from pathlib import Path

import torch


def _latest_epoch_key(checkpoint):
    epoch_keys = [key for key in checkpoint if key.startswith("epoch_")]
    if not epoch_keys:
        raise KeyError("No epoch checkpoints were found in the checkpoint file.")
    return max(epoch_keys, key=lambda key: int(key.split("_")[-1]))


def load_swin_classifier(
    model_name,
    num_labels,
    checkpoint_file,
    device,
    epoch_key=None,
):
    """Load the PavAnalytics Swin classifier and its saved training checkpoint."""
    try:
        from transformers import AutoModelForImageClassification
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for Grad-CAM inference. "
            "Install the project requirements before running explainability."
        ) from exc

    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Classification checkpoint not found: {checkpoint_file}\n"
            "Train the classification model first, or pass --checkpoint with the path "
            "to an existing PavAnalytics checkpoint."
        )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

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
    model.eval()
    print(f"Loaded {epoch_key} from {checkpoint_file}")
    return model, epoch_key


def get_swin_gradcam_target_layer(model):
    """Return the final spatial token normalisation layer used by Hugging Face Swin."""
    if not hasattr(model, "swin") or not hasattr(model.swin, "layernorm"):
        raise AttributeError(
            "Expected a Hugging Face Swin image-classification model with "
            "model.swin.layernorm available for Grad-CAM."
        )
    return model.swin.layernorm
