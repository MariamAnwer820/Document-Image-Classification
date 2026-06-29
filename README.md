# Document Image Classification Using Lightweight CNNs with OCR Fusion

A multimodal document image classification framework that combines lightweight convolutional neural networks with an OCR-based text classification pipeline, achieving **87.57% accuracy** on the RVL-CDIP benchmark through stacking meta-learner fusion — a **12.17 pp improvement** over the best single CNN model.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Models](#models)
- [Fusion Strategies](#fusion-strategies)
- [Robustness Evaluation](#robustness-evaluation)
- [Dataset](#dataset)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

Document image classification is a critical task in automated document processing pipelines, with applications spanning digital archiving, regulatory compliance, and enterprise content management. However, relying solely on visual features limits classification accuracy, particularly for document categories that are primarily distinguished by their textual content rather than visual layout.

This project addresses three key challenges:

1. **Computational constraints** — Many deployment scenarios involve edge devices or servers with limited budgets, necessitating lightweight models that maintain competitive accuracy.
2. **Visual-only brittleness** — Purely visual classifiers are brittle under common document perturbations such as rotation, noise, and blur, all of which occur frequently in real-world scanning environments.
3. **Visually similar categories** — Document categories that share similar layouts but differ in textual content (e.g., scientific reports vs. scientific publications, memos vs. letters) are notoriously difficult for vision-only approaches.

Our solution is a **multimodal fusion framework** that combines:
- A **CNN branch** with three architectures of increasing complexity (CustomCNN, MobileNetV2, EfficientNet-B0)
- An **OCR branch** with Tesseract text extraction, SBERT semantic embeddings, and multi-crop EfficientNet-B0 visual features
- **Eight fusion strategies** spanning score-level, feature-level, and meta-learning approaches

---

## Key Results

| System | Test Accuracy | Macro F1 |
|---|:---:|:---:|
| CustomCNN (scratch) | 55.19% | 0.5483 |
| MobileNetV2 (pretrained) | 68.50% | 0.6900 |
| EfficientNet-B0 (pretrained) | 75.44% | 0.7576 |
| Weighted Ensemble (3 CNNs) | 76.00% | 0.7618 |
| OCR Pipeline (SBERT + SVM + Specialists) | 76.46% | — |
| **Stacking Meta-Learner Fusion** | **87.57%** | — |

### Robustness Under Perturbation

| Perturbation | CNN Only | Fusion | Gain |
|---|:---:|:---:|:---:|
| 30° Rotation | 48.50% | 82.14% | +33.64 pp |
| Gaussian Noise (σ=50) | 56.75% | 82.50% | +25.75 pp |
| Gaussian Blur (σ=3) | 5.56% | 79.28% | +73.72 pp |
| Salt & Pepper (0.1) | 50.00% | 81.42% | +31.42 pp |

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RVL-CDIP Dataset (16 classes)               │
│                    16,000 images (1,000 per class)                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Preprocessing       │
                    │   Deskew + CLAHE      │
                    └───────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
       ┌────────┴────────┐            ┌─────────┴─────────┐
       │   CNN Branch    │            │    OCR Branch      │
       │                 │            │                    │
       │ • CustomCNN     │            │ • Tesseract OCR    │
       │ • MobileNetV2   │            │ • SBERT Embeddings │
       │ • EfficientNet  │            │ • Multi-crop Vis.  │
       │                 │            │ • SVM + Specialists │
       └────────┬────────┘            └─────────┬─────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Fusion Strategies   │
                    │                       │
                    │ • Score-Level (α)     │
                    │ • Class-Dependent     │
                    │ • Heatmap-Guided      │
                    │ • Feature-Level       │
                    │ • Stacking            │
                    │ • Post-Fusion Spec.   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Final Prediction    │
                    │   87.57% Accuracy     │
                    └───────────────────────┘
```

---

## Repository Structure

```
.
├── env.example                          # Unified environment configuration
├── fusion_notebook.ipynb                # Main fusion experiments notebook
├── fusion_notebook_CustomCNN.ipynb      # CustomCNN fusion variant
├── fusion_notebook_fixed.ipynb          # Fusion notebook (bug-fixed)
├── fusion_FUTURE_WORK.ipynb             # Ablation & future work cells
├── fusion_NEW_CELLS_ablation.ipynb      # Ablation study additions
├── fusion_NEW_CELLS_robustness.ipynb    # Robustness evaluation additions
├── build_notebook.py                    # Notebook builder script
├── generate_figures.py                  # Figure generation script
│
├── figures/                             # CNN experiment figures
│   ├── architecture_diagram.png
│   ├── confusion_matrix_effnet.png
│   ├── model_comparison.png
│   ├── per_class_f1_comparison.png
│   ├── training_curves_smooth.png
│   ├── transfer_learning_gap.png
│   ├── ablation_augmentation_f1.png
│   ├── ensemble_gains.png
│   ├── preprocessing_pipeline.png
│   └── model_diagram_mobilenetv2.png
│
├── report_charts_v2/                    # Report-quality figures
│   ├── architecture_diagram.png
│   ├── fusion_grand_comparison.png
│   ├── fusion_robustness_gain.png
│   ├── ocr_improvement.png
│   ├── per_class_f1_fusion.png
│   ├── robustness_2x2.png
│   ├── robustness_top.png
│   ├── robustness_bottom.png
│   └── alpha_search.png
│
├── report/                              # Generated report assets
│
├── Document_Image_Classification_Using_CNN-OCR_Fusion_Updated.pptx
├── Final_Report_Document_Image_Classification_Complete.docx
├── IEEE_Final_Report.docx
├── IEEE_Final_Report.pdf
└── README.md
```

> **Note:** The CNN training and OCR pipeline notebooks are maintained separately and should be placed alongside the fusion notebooks. Copy `env.example` to `.env` and configure paths before running any notebook.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Tesseract OCR installed on your system

### 1. Install Tesseract

**Windows:**
```bash
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Default path: C:\Users\<name>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS (Homebrew):**
```bash
brew install tesseract
```

### 2. Clone and Install Dependencies

```bash
git clone <repository-url>
cd document-image-classification

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp env.example .env
```

Edit `.env` and update the required paths:
- `SPLIT_OUTPUT_DIR` / `DATA_DIR` — path to the RVL-CDIP dataset
- `TESS_PATH` — path to Tesseract executable
- `FUSION_DIR` — path for fusion data

---

## Configuration

All parameters are managed through a single `.env` file (see `env.example`). The configuration is organized into 22 sections:

| Section | Description |
|---|---|
| 1. Global Settings | Seed, device, number of classes |
| 2. Paths | All data, model, and output directories |
| 3. Dataset | Samples per class limits |
| 4. Image Preprocessing | CLAHE, adaptive threshold, morphology |
| 5. Normalization | ImageNet vs. custom channel stats |
| 6. Data Augmentation | ColorJitter, Affine, RRC, Erasing |
| 7. MixUp | CustomCNN-only augmentation |
| 8. Test-Time Augmentation | TTA rounds and ranges |
| 9–12. CNN Training | LR, epochs, freeze, SWA |
| 13–14. Model Architecture | CustomCNN and pretrained heads |
| 15. Hard Class Boost | Per-class weight multipliers |
| 16. Ensemble & Ablation | Ensemble method, ablation config |
| 17. OCR Pipeline | Tesseract, SBERT, specialists |
| 18. Fusion | Fusion data paths |
| 19–21. Logging & Analysis | DPI, thresholds, CUDA |
| 22. Requirements | Python package dependencies |

### Key Parameters

```bash
# Must update before first run
SPLIT_OUTPUT_DIR=/path/to/rvlcdip/CNN_preprocessed
DATA_DIR=/path/to/rvlcdip/CNN_preprocessed
TESS_PATH=/path/to/tesseract
FUSION_DIR=/path/to/fusion

# Quick experiment defaults
NUM_CLASSES=16
BATCH_SIZE=64
IMG_SIZE=224
RANDOM_SEED=42
```

### Loading Configuration in Python

```python
from dotenv import load_dotenv
load_dotenv()

import os
DATA_DIR  = os.environ["DATA_DIR"]
IMG_SIZE  = int(os.getenv("IMG_SIZE", 224))
USE_TTA   = os.getenv("USE_TTA") == "True"
HEAD_FC   = os.getenv("HEAD_FC_SIZES").split(",")
```

---

## Usage

### 1. CNN Training & Evaluation

Open the CNN experiment notebook and run all cells:

```bash
jupyter notebook cnn_notebook.ipynb
```

The notebook will:
1. Load and preprocess the RVL-CDIP subset (Deskew + CLAHE)
2. Train CustomCNN, MobileNetV2, and EfficientNet-B0 sequentially
3. Evaluate each model on the test set with 95% Wilson confidence intervals
4. Generate the weighted ensemble predictions
5. Run the augmentation ablation study (MobileNetV2 without augmentation)
6. Produce per-class F1 analysis, confusion matrices, and training curves
7. Save model weights to `MODEL_EXPORT_DIR` and figures to `FIGURES_DIR`

**Expected training times** (NVIDIA T4 / equivalent):
- CustomCNN: ~30 min (30 epochs)
- MobileNetV2: ~20 min (25 epochs)
- EfficientNet-B0: ~35 min (40 epochs)

### 2. OCR Pipeline

Open the OCR notebook:

```bash
jupyter notebook ocr_notebook.ipynb
```

The notebook will:
1. Extract text from all document images using Tesseract OCR
2. Generate SBERT semantic embeddings (`all-mpnet-base-v2`, 768-dim)
3. Extract multi-crop visual features using EfficientNet-B0 (header + full page + footer → 3840-dim)
4. Combine and reduce features (2048-dim final representation)
5. Train a linear SVM with hyperparameter grid search (3-fold CV)
6. Train 14 specialist binary classifiers for the most confused class pairs
7. Apply confidence-based override (threshold = 0.70)
8. Cache all OCR results and embeddings for reuse

### 3. Fusion

Open the fusion notebook:

```bash
jupyter notebook fusion_notebook.ipynb
```

The notebook will:
1. Load CNN predictions (probability vectors) and OCR predictions
2. Align the two branch outputs on the shared test set
3. Evaluate all eight fusion strategies
4. Run the alpha optimization study for score-level fusion
5. Conduct the full robustness evaluation (rotation, noise, blur, salt & pepper)
6. Generate comparison charts and save results

---

## Models

### CustomCNN (From Scratch)

A lightweight 3-block CNN trained entirely from scratch, serving as a baseline to quantify the transfer learning gap.

| Attribute | Value |
|---|---|
| Parameters | 0.33M |
| Pretraining | None |
| Architecture | Conv blocks (32→64→128) + GAP + FC(128-256-16) |
| Augmentation | MixUp (α=0.2) |
| Test Accuracy | 55.19% |
| Macro F1 | 0.5483 |

### MobileNetV2 (Pretrained)

Depthwise separable convolutions with inverted residuals and linear bottlenecks, pretrained on ImageNet.

| Attribute | Value |
|---|---|
| Parameters | 4.07M |
| Pretraining | ImageNet |
| Head | FC(1280-1024-512-16) |
| Test Accuracy | 68.50% |
| Macro F1 | 0.6900 |

### EfficientNet-B0 (Pretrained)

Compound scaling with squeeze-and-excitation attention, pretrained on ImageNet — the best single CNN model.

| Attribute | Value |
|---|---|
| Parameters | 5.86M |
| Pretraining | ImageNet |
| Head | FC(1280-1024-512-16) |
| Test Accuracy | 75.44% |
| Macro F1 | 0.7576 |

### Weighted Ensemble

Accuracy-weighted average of softmax probabilities from all three CNN models.

| Attribute | Value |
|---|---|
| Test Accuracy | 76.00% |
| Macro F1 | 0.7618 |

---

## Fusion Strategies

We evaluate eight strategies for combining CNN and OCR predictions:

| # | Strategy | Type | Description |
|---|---|---|---|
| 1 | Score-Level Fusion | Score | Weighted average: `α·p_cnn + (1-α)·p_ocr` |
| 2 | Class-Dependent α | Score | Separate α per class |
| 3 | Heatmap-Guided | Score | Sample-adaptive α based on confidence gap |
| 4 | Feature-Level Fusion | Feature | Concatenated embeddings classified by SVM |
| 5 | Stacking Meta-Learner | Meta | Second-level classifier on 32-dim prediction vectors |
| 6 | Post-Fusion Specialists | Meta | Binary specialists applied to fusion output |
| 7 | Hybrid Strategy | Hybrid | Score-level fusion on full test set |
| 8 | OCR-Only Baseline | Baseline | OCR pipeline without CNN |

### Alpha Optimization

The optimal fusion weight `α` for score-level fusion is determined by grid search over the validation set. The stacking meta-learner implicitly learns the optimal combination weights, which is why it outperforms all fixed-α strategies.

---

## Robustness Evaluation

We evaluate model robustness under four perturbation types at multiple severity levels:

| Perturbation | Levels Tested | CNN Degradation | Fusion Resilience |
|---|---|---|---|
| Rotation | 5°, 15°, 30° | Severe (48.50% at 30°) | Strong (82.14% at 30°) |
| Gaussian Noise | σ = 25, 50, 75 | Moderate (56.75% at σ=50) | Strong (82.50% at σ=50) |
| Gaussian Blur | σ = 1, 2, 3 | Catastrophic (5.56% at σ=3) | Strong (79.28% at σ=3) |
| Salt & Pepper | 0.02, 0.05, 0.1 | Severe (50.00% at 0.1) | Strong (81.42% at 0.1) |

**Key insight:** The OCR branch provides a complementary signal that is largely unaffected by visual perturbations, making the fusion system dramatically more robust than the CNN alone. Under Gaussian blur where the CNN collapses to 5.56%, the fusion system still achieves 79.28% — a **73.72 percentage-point gain**.

---

## Dataset

### RVL-CDIP

The **Ryerson Vision Lab Complex Document Information Processing** dataset is the standard benchmark for document image classification.

| Property | Value |
|---|---|
| Total Images | 400,000 |
| Classes | 16 |
| Format | Grayscale |
| Source | IIT-CDIP collection (legacy tobacco litigation documents) |

**16 Document Categories:**
Advertisement, Budget, Email, File Folder, Form, Handwritten, Invoice, Letter, Memo, News Article, Presentation, Questionnaire, Resume, Scientific Publication, Scientific Report, Specification

### Our Subset

We use a balanced subset of 1,000 images per class (16,000 total) split 80/10/10:

| Split | Per Class | Total | Percentage |
|---|:---:|:---:|:---:|
| Training | 800 | 12,800 | 80% |
| Validation | 100 | 1,600 | 10% |
| Test | 100 | 1,600 | 10% |

### Tobacco-3482 (Optional)

An additional smaller dataset (3,482 images, 10 classes) for supplementary evaluation.

---

## Citation

If you use this work, please cite:

```bibtex
@article{attia2025docclass,
  title={Document Image Classification Using Lightweight CNNs with OCR Fusion},
  author={Attia, Rania and Salem, Nourhan and Youssef, Mariam},
  year={2025}
}
```

### Key References

1. Harley, A.W., Ufkes, A., Derpanis, K.G. (2015). *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval.* ICDAR.
2. Afzal, M.Z., et al. (2017). *Deepdocclassifier: Document classification with deep convolutional neural network.* ICIP.
3. Das, A., et al. (2018). *Document image classification with intra-domain transfer learning and stacked GNN representations.* ICDAR.
4. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
5. Tan, M., Le, Q.V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
6. Li, Y., et al. (2022). *DiT: Self-supervised Pre-training for Document Image Transformer.* ICDAR.
7. Appalaraju, S., et al. (2021). *DocFormer: Document Transformer with Spatial Features.* ICCV.
8. Tensmeyer, C., Martinez, T. (2017). *Analysis of Convolutional Neural Networks for Document Image Classification.* DAS.
9. Borchmann, L., et al. (2021). *Albert: Data augmentation for document understanding.* ACL.
10. Dauphin, Y., et al. (2019). *Diversity in document classification ensembles.* DAS.

---

## Authors

- **Rania Attia** — Led the data collection, sampling, and preprocessing pipeline for both the CNN and OCR branches, and subsequently led the fusion framework design and implementation, including all eight fusion strategies, the alpha optimization study, and the robustness evaluation. She contributed to the integration of CNN and OCR predictions and designed the experimental protocol for comparing fusion strategies.
- **Nourhan Salem** — Was responsible for the CNN experiments, including the design and training of the CustomCNN, MobileNetV2, and EfficientNet-B0 models, the augmentation ablation study, the ensemble evaluation, and the test-time augmentation analysis.
- **Mariam Youssef** — Developed the OCR classification pipeline, including the Tesseract integration, SBERT embedding extraction, multi-crop EfficientNet-B0 feature engineering, SVM hyperparameter optimization, and the specialist binary classifier system for confused class pairs.

---

## License

This project is for academic and research purposes. The RVL-CDIP dataset is governed by its own license — please refer to the original dataset documentation for usage terms.
