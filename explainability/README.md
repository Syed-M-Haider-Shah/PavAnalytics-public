# PavAnalytics Grad-CAM Explainability

This module generates Grad-CAM explanations for the Swin Transformer pavement-condition classifier used in PavAnalytics.

## What it explains

The classifier predicts one of the five pavement-condition classes (`1` to `5`). By default, Grad-CAM explains the model's predicted class. A specific class can also be requested.

For the Hugging Face Swin classifier, the implementation hooks `model.swin.layernorm`, the final normalised spatial-token representation before global pooling and classification. The token activations are converted back to their square spatial grid, weighted by the class gradients, upsampled to the classifier input resolution, and overlaid on the image crop seen by the model.

## Checkpoint

By default, the module automatically uses the same repository-local checkpoint produced by classification training:

```text
classification/checkpoints/
└── all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth
```

The latest saved `epoch_*` is used unless `--epoch` is supplied.

## Single image

From the repository root:

```bash
python -m explainability.main --image path/to/image.jpg
```

To explain a particular pavement class:

```bash
python -m explainability.main --image path/to/image.jpg --class-label 3
```

## Folder of images

```bash
python -m explainability.main --input-dir path/to/images
```

## Outputs

Outputs are written to `explainability/outputs/` by default:

```text
<name>_model_input.png
<name>_gradcam_heatmap.png
<name>_gradcam_overlay.png
```

The overlay is generated on the resized and centre-cropped image actually supplied to the classifier. This avoids showing Grad-CAM outside the spatial field seen by the model.

## Smoke test

The Grad-CAM engine can be tested without downloading Swin by using a small local model with Swin-like `[B, 49, C]` token features:

```bash
python explainability/tests/smoke_test_gradcam.py path/to/image.jpg /tmp/gradcam_test.png
```

This checks the hooks, gradient capture, transformer-token reshaping, heatmap normalisation and image overlay. Full Swin inference additionally requires `transformers`, the pretrained model files and the trained PavAnalytics classification checkpoint.
