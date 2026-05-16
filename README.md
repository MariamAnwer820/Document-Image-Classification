# Document Image Classification

A two-notebook pipeline for classifying scanned document images into 10 categories using the RVL-CDIP dataset. The project covers end-to-end data preprocessing (quality auditing, cleaning, deskewing) and a dual-model benchmark comparing a CNN image classifier against an OCR-based text classifier.

---

## Dataset

**RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)**  
10 document classes: `ADVE`, `Email`, `Form`, `Letter`, `Memo`, `News`, `Note`, `Report`, `Resume`, `Scientific`

The dataset is split into `train` (70%) / `val` (15%) / `test` (15%) with up to 500 images per class per split. Images are expected in the following folder structure:

```
rvlcdip_sample/
├── all/
│   ├── ADVE/
│   ├── Email/
│   └── ...
├── train/
├── val/
└── test/
```

---

## Notebook 1 — Preprocessing

`preprocessing_notebook.ipynb` audits and cleans the raw dataset before training. The notebook:

1. **Creates train/val/test splits** — stratified sampling from a flat `all/` directory.
2. **Quality scanning** — detects empty images (>99.7% white pixels) and skewed/rotated scans across all splits; reports per-class statistics.
3. **Whitespace feature analysis** — validates white pixel ratio as a discriminative signal. Finds it strongly separates `Email` and `ADVE` from dense-text classes (`News`, `Scientific`). A logistic regression on whitespace and pixel features confirms this baseline.
4. **Empty image removal** — visualises and deletes blank images with a confirmation step before deletion.
5. **Deskewing** — detects and corrects rotation using the `deskew` library (handles up to ±90°, white background fill). Visualises before/after pairs.
6. **Image quality fixes** — addresses three common scan artefacts:
   - JPEG blur → unsharp mask sharpening
   - Small/faint text → CLAHE local contrast enhancement
   - Low contrast → adaptive thresholding blended with the original

All cleaning steps produce summary charts saved as `.png` files.

---

## Notebook 2 — Classification

`document_classification_complete.ipynb` trains and compares two classifiers.

### Pipeline A — CNN (MobileNetV2)

- Pretrained on ImageNet; final classifier replaced with a custom head (`Dropout → Linear(256) → ReLU → Dropout → Linear(num_classes)`)
- Training augmentation: random horizontal flip, ±10° rotation, colour jitter
- Optimiser: Adam with weight decay; StepLR scheduler (halves LR every 4 epochs)
- Best model checkpointed by validation accuracy

### Pipeline B — OCR-MLP (PyTorch)

- Tesseract OCR (PSM 3, OEM 3) extracts text from each image; results cached to `ocr_cache/`
- Text cleaned (lowercased, single-char tokens removed, non-alphanumeric stripped)
- TF-IDF vectorisation (unigrams + bigrams, 30,000 features)
- 3-layer PyTorch MLP: `TF-IDF → 1024 → 256 → num_classes` with BatchNorm and Dropout
- Optimiser: Adam; ReduceLROnPlateau scheduler

### Evaluation

- Side-by-side accuracy bar chart
- Normalised confusion matrices for both models
- Full classification reports saved to `results/`

---

## Preprocessed Data

The cleaned dataset produced by the preprocessing notebook is available in the [`rvlcdip_preprocessed1`](https://github.com/MariamAnwer820/Document-Image-Classification/tree/main/rvlcdip_preprocessed1) folder. It contains the final `train/`, `val/`, and `test/` splits after empty image removal, deskewing, and quality fixes. Point `DATA_DIR` in the classification notebook to this folder.

---

## Setup

### Requirements

```bash
pip install torch torchvision pytesseract opencv-python scikit-learn \
            tqdm matplotlib seaborn pillow deskew scikit-image
```

**Tesseract OCR** must be installed separately:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

### Configuration

In `preprocessing_notebook.ipynb`:
```python
DATA_ROOT = r'path/to/rvlcdip_sample'
```

In `document_classification_complete.ipynb`:
```python
DATA_DIR = r'path/to/rvlcdip_preprocessed1'

# Windows only — set Tesseract path if not on PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Run Order

```
1. preprocessing_notebook.ipynb            # clean and prepare the dataset
2. document_classification_complete.ipynb  # train and evaluate models
```

---

## Outputs

| Path | Contents |
|---|---|
| `results/class_distribution.png` | Class counts per split |
| `results/cnn_training_curves.png` | CNN loss and accuracy over epochs |
| `results/mlp_training_curves.png` | MLP loss and accuracy over epochs |
| `results/model_comparison.png` | Val vs test accuracy bar chart |
| `results/model_comparison.csv` | Numeric summary |
| `results/cm_cnn.png` | CNN confusion matrix |
| `results/cm_mlp.png` | OCR-MLP confusion matrix |
| `results/classification_reports.txt` | Per-class precision, recall, F1 |
| `models/best_cnn.pth` | Best CNN checkpoint |
| `models/best_mlp.pth` | Best MLP checkpoint |
| `ocr_cache/` | Cached OCR JSON files (delete to force re-extraction) |
| `whitespace_feature.png` | Whitespace ratio distribution |
| `quality_fixes_preview.png` | Before/after for blur, contrast, and small text fixes |
| `rotation_before_after.png` | Before/after deskewing |

---

## Dependencies

| Library | Use |
|---|---|
| PyTorch / torchvision | CNN and MLP training |
| pytesseract | OCR text extraction |
| OpenCV | Image quality metrics and fixes |
| scikit-learn | TF-IDF, label encoding, metrics |
| deskew / scikit-image | Skew detection and correction |
| Pillow | Image I/O and augmentation |
| matplotlib / seaborn | Visualisations |
