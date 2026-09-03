import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def prepare_image(image_path, image_size=224):
    """Prepare an image exactly like the classification validation/test pipeline."""
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    spatial_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])
    display_image = spatial_transform(image)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    input_tensor = tensor_transform(display_image).unsqueeze(0)
    return display_image, input_tensor


def _extract_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise TypeError("Could not extract logits from the model output.")


def _normalise_cam(cam):
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    denom = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return cam / denom


def _tokens_to_spatial_cam(activations, gradients):
    """Convert [B, tokens, channels] transformer features to a 2D Grad-CAM."""
    if activations.ndim != 3 or gradients.ndim != 3:
        raise ValueError("Expected 3D token activations and gradients.")

    batch_size, num_tokens, _ = activations.shape
    side = int(math.sqrt(num_tokens))

    if side * side != num_tokens:
        side_without_first = int(math.sqrt(num_tokens - 1))
        if side_without_first * side_without_first == num_tokens - 1:
            activations = activations[:, 1:, :]
            gradients = gradients[:, 1:, :]
            num_tokens -= 1
            side = side_without_first
        else:
            raise ValueError(
                f"Cannot reshape {num_tokens} transformer tokens into a square spatial map."
            )

    weights = gradients.mean(dim=1, keepdim=True)
    cam = (activations * weights).sum(dim=-1)
    cam = torch.relu(cam)
    return cam.reshape(batch_size, side, side)


def _feature_map_to_cam(activations, gradients, expected_channels=None):
    """Support both token features and conventional 2D feature maps."""
    if activations.ndim == 3:
        return _tokens_to_spatial_cam(activations, gradients)

    if activations.ndim != 4:
        raise ValueError(
            "Grad-CAM target layer must return either [B, tokens, channels], "
            "[B, channels, H, W], or [B, H, W, channels]."
        )

    if expected_channels is not None and activations.shape[1] == expected_channels:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (activations * weights).sum(dim=1)
    elif expected_channels is not None and activations.shape[-1] == expected_channels:
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (activations * weights).sum(dim=-1)
    else:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (activations * weights).sum(dim=1)

    return torch.relu(cam)


class GradCAM:
    """Grad-CAM implementation that supports Swin token representations."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(tensor):
            raise TypeError("Grad-CAM target layer did not return a tensor.")
        self.activations = tensor

        if tensor.requires_grad:
            tensor.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients

    def generate(self, input_tensor, class_index=None):
        """Generate a Grad-CAM map for the predicted class or a supplied class index."""
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        logits = _extract_logits(output)

        if logits.ndim != 2:
            raise ValueError(f"Expected classification logits [B, C], got {tuple(logits.shape)}")
        if logits.shape[0] != 1:
            raise ValueError("This Grad-CAM inference helper currently expects batch size 1.")

        predicted_index = int(logits.argmax(dim=1).item())
        if class_index is None:
            class_index = predicted_index

        if not 0 <= int(class_index) < logits.shape[1]:
            raise ValueError(
                f"class_index must be between 0 and {logits.shape[1] - 1}, got {class_index}."
            )

        score = logits[:, int(class_index)].sum()
        score.backward()

        if self.activations is None:
            raise RuntimeError("No target-layer activations were captured during the forward pass.")
        if self.gradients is None:
            raise RuntimeError("No target-layer gradients were captured during backpropagation.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        classifier = getattr(self.model, "classifier", None)
        expected_channels = getattr(classifier, "in_features", None)
        cam = _feature_map_to_cam(activations, gradients, expected_channels)
        cam = _normalise_cam(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        probabilities = torch.softmax(logits.detach(), dim=1)
        confidence = float(probabilities[0, predicted_index].item())

        return {
            "cam": cam[0].cpu().numpy(),
            "predicted_index": predicted_index,
            "target_index": int(class_index),
            "confidence": confidence,
            "logits": logits.detach().cpu(),
        }

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_heatmap(cam):
    """Convert a normalised Grad-CAM array to an RGB heatmap."""
    cam_uint8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(display_image, cam, alpha=0.45):
    """Overlay Grad-CAM on the exact image crop seen by the classifier."""
    base = np.asarray(display_image.convert("RGB"), dtype=np.float32)
    heatmap = create_heatmap(cam).astype(np.float32)

    if heatmap.shape[:2] != base.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = (1.0 - alpha) * base + alpha * heatmap
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))


def save_gradcam_outputs(display_image, cam, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap = Image.fromarray(create_heatmap(cam))
    overlay = overlay_heatmap(display_image, cam)

    input_path = output_dir / f"{stem}_model_input.png"
    heatmap_path = output_dir / f"{stem}_gradcam_heatmap.png"
    overlay_path = output_dir / f"{stem}_gradcam_overlay.png"

    display_image.save(input_path)
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)

    return {
        "model_input": input_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
    }
