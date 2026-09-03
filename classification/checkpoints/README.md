# Classification checkpoints

`run_classification.py` stores the trained Swin checkpoint in this directory by default:

```text
all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth
```

PyTorch checkpoint files (`*.pth`) are excluded by the repository `.gitignore`. Supply a checkpoint explicitly with `--checkpoint` when running test or Grad-CAM modes if it is stored elsewhere.
