# Diabetic Retinopathy Screening Project

**Quality-Aware, Cross-Dataset, Explainable Deep Learning Framework for Automated Diabetic Retinopathy Screening**

An end-to-end pipeline that unifies four public retinal fundus datasets, trains and compares a series of CNN and multi-branch fusion architectures for 5-class Diabetic Retinopathy (DR) grading, and adds a full explainability, clinical evaluation, and fine-tuning layer on top of the best model — with Grad-CAM lesion localization, cross-modal attention analysis, and statistically-validated error diagnostics.

---

## Overview

Diabetic Retinopathy is graded on a 5-point severity scale:

| Grade | Class |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR (PDR) |

This project builds a screening classifier for that scale by merging four heterogeneous public datasets into one standardized pipeline, then progressively improving the model — from a single CNN baseline to a learned cross-modal attention fusion of two backbones, fine-tuned end-to-end and validated with clinical-grade metrics and explainability.

## Dataset

Images and labels are merged from four sources into a single unified metadata table (~42,500 images):

| Source | Images | Notes |
|---|---|---|
| **EyePACS** | ~35,100 | Largest source; primary driver of class imbalance |
| **APTOS 2019** | ~5,590 | Kaggle competition set |
| **Messidor-2** | ~1,744 | Clinically curated |
| **IDRiD** | ~81 | Smallest, expert-annotated with lesion masks |

Each image is tracked through `image_key`, `dataset`, `processed_path`, `label_dr_standard` (0–4), `eligible_for_model`, and `model_split` (train/val/test) in the unified metadata CSVs.

> **Two metadata files exist** — `unified_metadata.csv` (initial unified split) and `corrected_unified_metadata.csv` (the corrected train/val/test split that every trained model checkpoint was actually built and evaluated against, from Phase 9 onward). **Use `corrected_unified_metadata.csv`** for anything involving the trained checkpoints.

Class distribution is heavily imbalanced toward "No DR" (Grade 0), which is addressed via class-weighted loss functions throughout.

## Pipeline

| Phase | Notebook | What it does |
|---|---|---|
| 1 | `01_dataset_inspection.ipynb` | Raw dataset inspection across all 4 sources |
| 2 | `02_label_standardization.ipynb` | Unifies each dataset's label scheme onto the common 0–4 grade |
| 3 | `03_image_quality_analysis.ipynb` | Brightness/contrast/sharpness scoring, quality flagging |
| 4 | `04_duplicate_detection.ipynb` | Exact + perceptual-hash duplicate detection across datasets |
| 5 | `05_image_preprocessing.ipynb` | Standardizes all images into `data/processed/images/` |
| 6 | `06_phase3_metadata_enrichment.ipynb` | Adds DME labels, lesion annotation availability, patient-ID tracking |
| 7 | `07_eda_report.ipynb` | Exploratory analysis: class balance, per-dataset shift, quality distribution |
| 8 | `08_baseline_dr_classification.ipynb` | First baseline classification experiments |
| 9 | `09_model_building.ipynb` | Trains and benchmarks all model variants (see Results below) |
| 10 | `10_explainable-ai-using-grad-cam-heatmaps.ipynb` | Grad-CAM / Grad-CAM++ explainability across all branches |
| 11 | `11_comprehensive-evaluation-attention-dynamics-err.ipynb` | Full clinical evaluation suite, attention analysis, error diagnostics |
| 12 | `12_fine-tuned-joint-weighted-resnet-18-efficientnet.ipynb` | Joint fine-tuning of the attention fusion model with differential LRs |

## Models

| Model | Description |
|---|---|
| ResNet-18 Baseline | Plain transfer-learned ResNet-18 |
| EfficientNet-B0 | Plain transfer-learned EfficientNet-B0 |
| **Weighted ResNet-18** | ResNet-18 + class-weighted cross-entropy loss |
| ResNet18 + EfficientNet Naive Concatenation | Simple feature-concat fusion |
| ResNet18 + EfficientNet + Quality Fusion | Fusion augmented with image quality features |
| **Attention Fusion** | Learned cross-modal attention weighting ResNet-18 and EfficientNet-B0 branches per image |
| **Joint Fine-Tuned Attention Fusion** | Attention Fusion with the top backbone layers (`resnet.layer4`, `efficientnet.features[6:]`) unfrozen and fine-tuned jointly, using differential learning rates |

## Results (test set, `corrected_unified_metadata.csv`)

| Model | Accuracy | Macro F1 | Balanced Acc | QWK | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| ResNet-18 Baseline | 0.581 | 0.489 | 0.647 | 0.628 | 0.368 | 0.871 |
| EfficientNet-B0 | 0.549 | 0.445 | 0.538 | 0.584 | 0.311 | 0.829 |
| Naive Concatenation Fusion | 0.655 | 0.534 | 0.624 | 0.669 | 0.415 | 0.877 |
| Quality-Augmented Fusion | 0.649 | 0.532 | 0.630 | 0.661 | 0.412 | 0.876 |
| **Attention Fusion** | 0.733 | 0.691 | 0.797 | 0.742 | 0.560 | 0.940 |
| **Joint Fine-Tuned Attention Fusion** | **0.758** | **0.683** | 0.777 | **0.763** | 0.570 | — |

**Quadratic Weighted Kappa (QWK)** is the primary clinical metric (rewards near-miss grading, penalizes large errors) — the joint fine-tuned model reaches **0.763**, matching/exceeding the original single-CNN Weighted ResNet-18 baseline (0.763) while retaining interpretable per-branch attention weights.

> Note: a standalone re-evaluation of the Weighted ResNet-18 checkpoint in Phase 11 (`reports/tables/phase_11/computed_metrics_this_run.csv`) showed a lower accuracy (0.579) than its Phase 9 benchmark (0.743) on the same split — this discrepancy is unresolved and worth investigating (likely a checkpoint/threshold mismatch) before treating that specific number as final.

Full metric tables and figures: [`reports/tables/`](reports/tables/) and [`reports/figures/`](reports/figures/).

## Explainability & Evaluation Highlights

- **Grad-CAM / Grad-CAM++** heatmaps generated independently for the ResNet-18 branch, EfficientNet-B0 branch, and the attention-fused representation, with CLAHE-enhanced overlays to verify the model attends to actual lesions (microaneurysms, hemorrhages, exudates) rather than artifacts. See [`reports/figures/phase10_gradcam/`](reports/figures/phase10_gradcam/).
- **Cross-modal attention dynamics**: how much the model relies on each backbone, broken down per DR severity grade. See [`reports/tables/phase_11/attention_weights_summary.csv`](reports/tables/phase_11/attention_weights_summary.csv).
- **Clinical error taxonomy**: misclassifications are categorized into critical false negatives (referable DR missed), severe false positives, adjacent-grade discrepancies, and multi-grade discordance, each paired with a Grad-CAM overlay. See [`reports/tables/phase_11/clinical_error_analysis.csv`](reports/tables/phase_11/clinical_error_analysis.csv).
- **Statistical rigor**: 1000-iteration bootstrap confidence intervals and paired significance testing between model variants. See [`reports/tables/phase_11/bootstrap_statistical_comparison.csv`](reports/tables/phase_11/bootstrap_statistical_comparison.csv).

## Repository Structure

```
├── data/
│   └── metadata/                  # unified_metadata.csv, corrected_unified_metadata.csv
├── notebooks/                     # Phases 1–12, run in order
├── src/
│   ├── gradcam.py                 # Grad-CAM / Grad-CAM++ engine
│   ├── visualization.py           # CLAHE, heatmap overlays, explanation plots
│   ├── metrics.py                 # 11-metric clinical evaluation suite
│   ├── attention_analysis.py      # Cross-modal attention weight analysis
│   ├── error_analysis.py          # Clinical error taxonomy + failure case plots
│   ├── statistical_tests.py       # Bootstrap CIs, paired significance testing
│   └── phase1.py, phase2.py, ...  # Data pipeline modules (Phases 1–5)
├── reports/
│   ├── figures/                   # All generated plots, by phase
│   └── tables/                    # All generated metric/results CSVs, by phase
├── requirements.txt
└── README.md
```

> Trained model checkpoints (`.pth`) and raw/processed images are **not tracked in git** (see `.gitignore`) due to size — they're managed as Kaggle Datasets and referenced by path in the notebooks.

## Environment & Reproducing Results

Model training/inference notebooks (10–12) are designed to run on **Kaggle GPU** (T4/P100), using `torch.autocast` mixed precision and dynamic path resolution that works both on Kaggle and locally.

**Kaggle setup**, attach these datasets:
- Metadata: `corrected_unified_metadata.csv`
- Images: `processed/images/{aptos,eyepacs,idrid,messidor2}/`
- Checkpoints: `best_resnet18_weighted.pth`, `best_efficientnet_b0.pth`, `best_weighted_resnet_efficientnet_attention.pth`

**Local setup** (data pipeline / code editing only — training requires Kaggle):
```bash
pip install -r requirements.txt
```

**Run order:** notebooks 1 → 12 sequentially. Phases 1–9 build the dataset and baseline/fusion models; Phases 10–12 (explainability, evaluation, fine-tuning) require the Phase 9 checkpoints as input.

## Tech Stack

PyTorch · torchvision (ResNet-18, EfficientNet-B0) · scikit-learn · OpenCV (CLAHE, Grad-CAM overlays) · pandas/numpy · matplotlib/seaborn · Kaggle GPU (T4/P100, AMP mixed precision)


## Run the screening UI

Place `best_joint_finetuned_attention_fusion_repacked.pth` in the `models/` folder. Model checkpoints are not stored in Git; share the checkpoint separately with teammates.

In PowerShell, from the project root:

```powershell
py -m venv .venv-ui
.\.venv-ui\Scripts\Activate.ps1
python -m pip install -r .\requirements-ui.txt
streamlit run .\app.py