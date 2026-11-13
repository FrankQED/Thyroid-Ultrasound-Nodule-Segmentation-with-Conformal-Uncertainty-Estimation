# Red, Blue and Yellow: Mondrian Conformal Risk Control for Thyroid Nodule Segmentation in Ultrasound Images

Code, trained weights and test logits for the extended version of our CIBB 2026 paper.
A preliminary version of this work is available as arXiv:2610.13604.

We train DeepLabV3+ with an EfficientNet-B5 encoder on TN3K, with three channel-attention modules (SE, CBAM, GCA) and two losses (Dice-BCE and Boundary), and compare the eight configurations with paired tests on per-image Dice.
On top of the selected model (CBAM + Dice-BCE) we replace the pixel-level conformal layer of the preliminary version with image-level conformal risk control (CRC): for each test image the method returns an inner and an outer mask, whose radii are chosen so that the expected fraction of missed nodule and the expected fraction of deep false positives are below a level δ. Mondrian CRC repeats the calibration within strata of predicted nodule area.

## Contents

| Path | Content | Paper | Hardware |
|---|---|---|---|
| `notebooks/01_models_and_training.ipynb` | Data, models, losses, training, evaluation, export of the test masks (CPU) and logits (GPU) | Tables 1-2 | GPU |
| `notebooks/02_conformal_v1_pixel.ipynb` | Pixel-level split conformal of the preliminary version, and why it is replaced | Introduction | CPU |
| `notebooks/03_conformal_v2_crc.ipynb` | Image-level CRC, Mondrian CRC, robustness checks | Tables 4-6, Figure 2 | CPU |
| `notebooks/04_model_comparison.ipynb` | Paired tests on per-image Dice (centred bootstrap, Holm correction), qualitative comparison | Table 3, Figure 1 | CPU |
| `scripts/prepare_release.py` | Builds the release files (removes the optimizer state from the checkpoints, writes SHA-256 checksums) | | |
| `scripts/download_assets.py` | Downloads weights and logits from the release and verifies the checksums | | |
| `figures/` | Figures produced by notebooks 02-04 | | |
| `results/` | CSV outputs of notebook 04 | | |

Notebooks 02-04 are saved with their outputs, computed on the released logits.

## Weights and logits

Weights and logits are attached to the GitHub release [`v1.0`](https://github.com/FrankQED/Thyroid-Ultrasound-Nodule-Segmentation-with-Conformal-Uncertainty-Estimation/releases/tag/v1.0) (files above 100 MB cannot be stored in the repository itself).

| File | Content |
|---|---|
| `best_<loss>_<model>.pt` (8 files, about 119 MB each) | Weights of the eight configurations (`model_state_dict`), with epoch, best validation Dice and training configuration |
| `test_<attention>_<loss>.npy` (8 files, 161 MB each) | Raw logits on the 614 TN3K test images, float32, 614 x 256 x 256 |
| `test_names.txt` | File names of the test images, in the same order as the logits |
| `SHA256SUMS.txt` | SHA-256 of every file and the path where the notebooks expect it |

`python scripts/download_assets.py --what all` downloads everything into `checkpoints/` and `cache/logits/` and checks the checksums.

The ground-truth masks are not redistributed, since they are the TN3K annotations: notebook 01 rebuilds them from the dataset as `cache/logits/test_masks.npy` (Section 8A, CPU only, no checkpoint needed). They are needed by notebooks 02-04.

Checkpoint names come from the class names: `best_DiceBCELoss_CBAM_DeepLabV3Plus.pt` is CBAM + Dice-BCE, `best_Boundary_Loss_DeepLabV3Plus.pt` is the model without attention trained with the Boundary loss, and so on (full list in `CONFIGS`, notebook 01).
The logits of a model give its predicted mask with `logits > 0` (probability > 0.5).

## Data

TN3K is distributed by its authors through the [TRFE-Net repository](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation); we do not redistribute the images.
Download `Thyroid Dataset.zip` and place it in `data/` inside the project folder (below). Notebook 01 extracts it to `data/Thyroid Dataset/tn3k/` (folders `trainval-image`, `trainval-mask`, `test-image`, `test-mask`).

We use the original TN3K train/test split. The 2879 training images are split into training and validation (85%/15%) with seed 42. The 614 test images are used only for evaluation and for the conformal analyses.

## Project folder

All notebooks read and write the same folder:

```
PROJECT_ROOT/
    data/Thyroid Dataset/tn3k/   TN3K images and masks
    checkpoints/                 trained weights
    cache/logits/                test logits, masks and file names
```

- Locally, `PROJECT_ROOT` is the root of this repository (`data/`, `checkpoints/` and `cache/` are ignored by git). Set the environment variable `PROJECT_ROOT` to use another folder.
- On Google Colab, notebook 01 mounts Drive and uses `MyDrive/thyroid_seg`.
- Notebooks 02-04 read the logits from `PROJECT_ROOT/cache/logits`, or from the folder in the environment variable `THYROID_CACHE`.

## Reproducing the results

### Option A: analyses from the released logits (CPU)

Reproduces Tables 3-6, the conformal results and Figures 1-2.

1. Download TN3K (see Data) and the logits: `python scripts/download_assets.py --what logits`.
2. Build the ground-truth masks: in notebook 01 run Sections 1 and 2, then Section 8A (CPU, a few minutes; needs `requirements-gpu.txt` for the data pipeline).
3. Run notebooks 02, 03 and 04 (`pip install -r requirements.txt`).

The first cell of notebooks 02 and 03 prints the global Dice of CBAM + Dice-BCE, which must be 89.60 (Table 1).

### Option B: evaluation from the released weights (GPU)

Reproduces Tables 1-2 and regenerates the logits.

1. Download TN3K (see Data) and the weights: `python scripts/download_assets.py --what checkpoints`.
2. In notebook 01 run Sections 1-4 and 6, then Section 7 (evaluation of the eight models) and Section 8 (export of masks and logits). Section 8B prints the global Dice of each model, to compare with Table 1.

### Option C: full retraining (GPU)

1. In notebook 01 run Sections 1-4, set `ATTENTION` and `LOSS` in Section 5 and run the training. Repeat for the eight combinations.
2. Run Sections 6-8 as in option B, then notebooks 02-04.

Training setup: at most 100 epochs, batch size 4, input 256 x 256, AdamW (learning rate 1e-4, weight decay 1e-4), cosine annealing down to 1e-6, early stopping on the validation Dice with patience 30. The checkpoint with the best validation Dice is kept.
Retraining on a different GPU or library version gives slightly different weights, because cuDNN convolutions are not deterministic; the released weights are the ones used in the paper.

### Seeds

| Step | Seed |
|---|---|
| Training (Python, NumPy, PyTorch) and train/validation split | 42 |
| Calibration/evaluation split of the pixel-level conformal (notebook 02) | 42 |
| Calibration/evaluation splits for CRC (200 repetitions, notebook 03) | 0, ..., 199 |
| Centred bootstrap for the paired tests (10 000 resamples, notebook 04) | 0 |
| Figures with one calibration split | 42 |

## Environment

- Notebook 01: Google Colab with an NVIDIA T4, PyTorch provided by Colab, `segmentation-models-pytorch==0.3.4`, `albumentations==1.4.0` (`requirements-gpu.txt`).
- Notebooks 02-04: Python 3.12 with the versions in `requirements.txt`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the reference will be updated after publication) and the TN3K dataset:

- H. Gong et al., "Multi-task learning for thyroid nodule segmentation with thyroid region prior", IEEE ISBI 2021.
- H. Gong et al., "Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules", Computers in Biology and Medicine.

## License

The code in this repository is released under the MIT License (see `LICENSE`).

The weights and logits in the release are derived from TN3K. The images and the annotations are not redistributed. If you use these files, follow the terms of the TN3K dataset and cite its authors (see Citation).
The models are research artifacts and are not validated for clinical use.
