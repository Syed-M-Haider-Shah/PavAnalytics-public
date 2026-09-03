# PavAnalytics

Automating Pavement Condition Rating: A Deep Learning-Based Framework for Cycle Routes and Greenways.

## Overview

PavAnalytics is a deep learning framework designed to automatically assess pavement surface conditions from cycling infrastructure imagery.

The project includes:

- Fisheye Distortion Correction
- Road Surface Segmentation
- Pavement Condition Classification
- Explainable AI (Grad-CAM)
- Performance Evaluation

The released implementation is organised so that the three main workflows can be run independently: classification with training/testing/Grad-CAM, road segmentation, and fisheye preprocessing.

## Project Website

For additional project information, demonstrations, and results, visit:

Project Website: [https://www.paveanalytics.eu/](https://www.paveanalytics.eu/)

## Publications

### Conference Papers

- Shah, S. M. H., Qureshi, W. S., Dea, G. O. and Ullah, I. (2025). Intelligent Pavement Condition Rating System for Cycle Routes and Greenways. In Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS; ISBN 978-989-758-745-0; ISSN 2184-495X, SciTePress, pages 668-675. DOI: 10.5220/0013505000003941
- Garcia, J. A. A., Shah, S. M. H., Baig, M. H., Qureshi, W. S. and Ullah, I. (2025). Enhancing Pavement Condition Assessment: A Comprehensive Review of Affordable Sensing Technologies for Cycle Tracks. In Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS; ISBN 978-989-758-745-0; ISSN 2184-495X, SciTePress, pages 676-682. DOI: 10.5220/0013505100003941
- M. H. Baig, J. A. Ayala Garcia, W. S. Qureshi, I. Ullah, Towards assessing cycleway pavement surface roughness using an action camera with imu and gps, in Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS, INSTICC, SciTePress, 2025, pp. 247–255. doi:10.5220/0013504900003941.

## Pavement Condition Rating Scale

We have introduced an intelligent pavement rating system for cycleways, named the Cycle Route Surface Index, a colour-coded, five-level rating system that combines visual inspection, roughness, vegetation, and drainage data to provide a clear and consistent pavement quality measure.

![Pavement Condition Rating Scale](figures/Rating-scale.jpg)

## Project Structure

```text
PavAnalytics/
├── classification/        # Swin Transformer classification training and evaluation
├── segmentation/          # SegFormer road segmentation and trained model configuration
├── explainability/        # Grad-CAM implementation for the pavement classifier
├── fisheye_correction/    # Checkerboard calibration and fisheye rectification
├── data/                  # Dataset/sample images already provided with the repository
├── figures/               # CRSI rating-scale and project figures
├── run_classification.py  # Train, test and generate Grad-CAMs for the test set
├── run_segmentation.py    # Batch road segmentation for a supplied folder
├── run_fisheye.py         # Detect and correct project-style fisheye frames in a folder
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or later is recommended. Create a virtual environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

The classification model uses `microsoft/swin-tiny-patch4-window7-224` from Hugging Face. Training also uses Weights & Biases (`wandb`) in the supplied experiment code.

## 1. Classification, Testing and Grad-CAM

`run_classification.py` is the main entry point for the pavement-condition classifier. It keeps the supplied Swin Transformer training logic but replaces machine-specific dataset arguments at the top-level runner with command-line paths.

The expected dataset structure is:

```text
Train/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/

Validation/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/

Test/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/
```

Run the full experiment:

```bash
python run_classification.py \
  --mode all \
  --train-dir /path/to/Train \
  --val-dir /path/to/Validation \
  --test-dir /path/to/Test
```

Available modes are:

```text
all       training -> validation -> testing -> Grad-CAM
test      evaluate an existing checkpoint
gradcam   generate Grad-CAM outputs for the test set
train     train and save the classification checkpoint
```

For test-only evaluation:

```bash
python run_classification.py \
  --mode test \
  --test-dir /path/to/Test \
  --checkpoint /path/to/classifier_checkpoint.pth
```

For Grad-CAM on the complete test set:

```bash
python run_classification.py \
  --mode gradcam \
  --test-dir /path/to/Test \
  --checkpoint /path/to/classifier_checkpoint.pth
```

Grad-CAM outputs are separated into correct and incorrect predictions and include the true and predicted CRSI classes in the output directory names.

The original supplied classification script is retained as `classification/original_classification_model.py` for reference. The modular files under `classification/` contain the same experiment components in reusable form.

## 2. Road Segmentation

The segmentation workflow uses a two-class SegFormer model to identify road/pavement pixels. Class `1` is treated as the road class, following the supplied segmentation implementation.

To segment every supported image in a folder:

```bash
python run_segmentation.py \
  --input-dir /path/to/images \
  --output-dir /path/to/segmentation_outputs
```

Use `--recursive` to process subdirectories:

```bash
python run_segmentation.py \
  --input-dir /path/to/images \
  --output-dir /path/to/segmentation_outputs \
  --recursive
```

For every input image, the runner saves:

```text
<name>_segment.png   # road pixels retained, background removed
<name>_mask.png      # binary road mask
<name>_overlay.png   # road prediction overlaid on the original image
```

The saved SegFormer processor configuration uses a 512 × 512 input. The trained model files are expected under:

```text
segmentation/checkpoints/final_model/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

## 3. Fisheye Distortion Correction

The fisheye module supports the checkerboard-calibration workflow used in the project: estimation of the intrinsic matrix `K` and distortion coefficients `D`, generation of rectification maps, and correction with OpenCV `cv2.remap`.

For folder-based preprocessing, `run_fisheye.py` first checks each image for the strong dark circular/oval border present in the PavAnalytics fisheye footage. Images that do not meet this signature are copied unchanged. This prevents normal pavement images from being unnecessarily transformed.

```bash
python run_fisheye.py \
  --input-dir /path/to/images \
  --output-dir /path/to/corrected_images
```

Use `--recursive` for subfolders. The runner writes `fisheye_processing_log.csv` describing the detection measurements and action applied to each image.

The automatic correction profile is currently defined for the validated 2048 × 1152 project-style fisheye frame and produces a 720 × 720 rectilinear output with a default 75° field of view. An automatically detected fisheye image with another resolution is left unchanged and recorded as an unsupported size rather than being corrected with inappropriate parameters.

The tuned profile is an empirical preprocessing profile. It is not presented as a replacement for measured checkerboard calibration. The calibration workflow in `fisheye_correction/calibration.py` and `fisheye_correction/correction.py` should be used when the camera's measured `K` and `D` parameters are available.

## Model Checkpoints

### Segmentation

The SegFormer model configuration and preprocessing configuration are stored under `segmentation/checkpoints/final_model/`. The trained `model.safetensors` file belongs in the same directory.

### Classification

Training saves the Swin classifier checkpoint to:

```text
classification/checkpoints/all_model_checkpoints_2layer_2e-5_retrained_all-road-surface dataset-2nd.pth
```

`.pth` files are excluded by `.gitignore` to avoid accidental publication of local experimental checkpoints. A different checkpoint can be supplied explicitly with `--checkpoint` when running `run_classification.py`.

## Technology Stack

### Core Libraries

| Library / Framework | Purpose |
| ------------------- | ------- |
| Python | Core programming language |
| PyTorch | Deep learning model development, training and inference |
| Hugging Face | Pretrained Swin Transformer and SegFormer model support |
| Transformers | Image classification and semantic segmentation architectures |
| OpenCV | Fisheye calibration, remapping and image processing |
| NumPy | Numerical computation and image-array processing |
| Pandas | Training-result and processing-log handling |
| Matplotlib | Training/evaluation visualisation |
| Scikit-learn | Classification metrics and confusion matrices |
| Pillow (PIL) | Image loading and output generation |
| Grad-CAM | Spatial explanation of CRSI classifier predictions |
| Weights & Biases | Experiment logging used by the supplied classifier training code |

## Reproducibility Notes

- The modular code preserves the supplied classification and segmentation logic while moving reusable operations into repository files and top-level runners.
- `classification/original_classification_model.py` and `segmentation/original_segmentation.py` retain the supplied scripts for comparison.
- Training, validation and test folders should remain independent.
- Grad-CAM should be generated from the same trained checkpoint used for test evaluation.
- The automatic fisheye detector is designed for the characteristic dark-border project footage and should not be interpreted as a general-purpose fisheye-lens classifier.

## Status

The repository now contains the implementation for pavement classification, test-set Grad-CAM generation, road segmentation and fisheye preprocessing. Additional datasets and large experiment artefacts are not required to be stored directly in the repository.
