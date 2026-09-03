# PavAnalytics Road Segmentation

This module applies the trained two-class SegFormer semantic-segmentation model to pavement images. Class `1` is treated as the road class, matching the supplied PavAnalytics segmentation code.

## Checkpoint

The trained model directory is:

```text
segmentation/checkpoints/final_model/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

The saved processor uses 512 × 512 input preprocessing.

## Batch runner

From the repository root:

```bash
python run_segmentation.py \
  --input-dir /path/to/images \
  --output-dir /path/to/outputs
```

Add `--recursive` to process images inside subfolders.

For each image, the runner writes a road-only image, binary mask and road-mask overlay.

## Original script

`original_segmentation.py` preserves the supplied segmentation script. `main.py` is the repository-relative modular entry point used before the folder-based top-level runner was added.
