# AI for Sustainability — SRIP 2026

## Earth Observation Pipeline: Delhi Airshed Land-Use Classification

**Disclosure:** AI tools (Antigravity / Google DeepMind) were used to assist in planning and writing this code. I fully understand the code I submit and am able to clearly explain it during the one-on-one discussion.

---

### Pipeline Overview

| Section | Task | Marks |
|---------|------|-------|
| Q1 | Spatial Reasoning & Data Filtering | 4 |
| Q2 | Label Construction & Dataset Preparation | 6 |
| Q3 | Model Training & Supervised Evaluation | 5 |

### Project Structure

```
AI_for_Sustainability/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── delhi_airshed_pipeline.ipynb   # Full pipeline (Q1 → Q2 → Q3)
├── data/                              # Downloaded at runtime (not committed)
│   ├── delhi_ncr_region.geojson
│   ├── delhi_airshed.geojson
│   ├── worldcover_bbox_delhi_ncr_2021.tif
│   └── rgb/                           # 9,216 Sentinel-2 patches
└── outputs/                           # Generated plots (committed to git)
    ├── q1_grid_overlay.png
    ├── q1_filtered_map.png
    ├── q2_class_distribution.png
    ├── q2_sample_images.png
    ├── q3_training_curves.png
    ├── q3_confusion_matrix.png
    ├── labels.csv
    ├── run_log.txt                    # Full pipeline log for analysis
    └── best_model.pth                 # Not committed (.gitignore)
```

### Setup & Reproduction

1. **Open** `notebooks/delhi_airshed_pipeline.ipynb` in Google Colab
2. **Set runtime** to **GPU → T4** (Runtime → Change runtime type)
3. **Upload `kaggle.json`** to the Colab files panel (📁 icon in sidebar)
4. **Run all cells** sequentially — the notebook installs deps, downloads data, and runs Q1 → Q2 → Q3

### Key Design Decisions

1. **Grid CRS:** EPSG:32644 (UTM Zone 44N) for metric grid construction, reprojected to EPSG:4326 for plotting
2. **Label extraction:** `rasterio.windows.Window` with `boundless=True` for safe boundary handling
3. **Mode computation:** `scipy.stats.mode` on non-zero pixels only (avoids nodata contamination)
4. **Split strategy:** 60/40 train-test per rubric; validation carved from the 60% training allocation (48% train / 12% val)
5. **Model:** ResNet-18 (ImageNet pretrained), FC head replaced for 5-class output
6. **Class imbalance:** Weighted `CrossEntropyLoss` (weights computed on train set only)
7. **Checkpoint:** Best model saved by validation F1-Macro (not accuracy)
8. **T4 GPU:** Mixed precision (`torch.cuda.amp`), `pin_memory=True`, `batch_size=64`

### Dataset

Downloaded automatically from [Kaggle](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed) (~302 MB).

> **Note on spatial leakage:** The assignment mandates a random split. Adjacent patches may land in different splits. This is a known limitation, accepted per rubric requirements.
