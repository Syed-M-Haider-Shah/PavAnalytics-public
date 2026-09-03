# PavAnalytics Classification

This directory contains the pavement-condition classification experiment supplied for PavAnalytics.

## Source preservation

`original_classification_model.py` is an untouched copy of the supplied complete classification script. The modular Python files organise that experiment into separate components. The repository-local checkpoint handling is the only intentional change to the original machine-specific checkpoint workflow.

## Structure

```text
classification/
├── __init__.py
├── config.py
├── dataset.py
├── transforms.py
├── model.py
├── train.py
├── evaluate.py
├── main.py
├── checkpoints/            # Created automatically when the model runs
├── original_classification_model.py
└── requirements.txt
```

- `config.py`: W&B experiment configuration, device selection and repository-relative output paths.
- `dataset.py`: `CustomImageDataset`.
- `transforms.py`: training and validation/test image transforms.
- `model.py`: Swin Transformer creation, optional local checkpoint reuse, weight verification, layer freezing and evaluation checkpoint loading.
- `train.py`: mixed-precision training loop and epoch result collection.
- `evaluate.py`: validation and test-set evaluation, classification report and confusion matrix.
- `main.py`: dataset loading, data loaders, optimiser, scheduler, loss, training, checkpoint reload, CSV output and final testing.

## Checkpoint behaviour

No absolute user-specific checkpoint path is required.

The checkpoint file is stored automatically at:

```text
classification/checkpoints/all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth
```

On a fresh clone:

1. If no local checkpoint exists, the model starts from `microsoft/swin-tiny-patch4-window7-224`.
2. Training saves the checkpoint inside `classification/checkpoints/`.
3. Before test evaluation, the model reloads `epoch_50` from that exact saved checkpoint file.

On a later run, if the same checkpoint file already exists, its latest epoch is automatically reused for the non-classifier model weights, matching the transfer behaviour of the original experiment. The new training run then writes to the same repository-local checkpoint path.

The checkpoint directory is created automatically, so the command can be run from a cloned repository without editing a Windows user path.

## Dataset paths

The supplied experiment still contains the original machine-specific dataset paths for the train, validation and test data. These have not been changed as part of the checkpoint update.

## Run

From the repository root:

```bash
python -m classification.main
```

The run requires access to the Hugging Face model `microsoft/swin-tiny-patch4-window7-224`, the pavement dataset directories and a working Weights & Biases configuration.
