# Stage 3 — Clustering

Read order-parameter MAT files from `../data/order_params/`, scale features,
and run an unsupervised clustering algorithm to label every molecule as
**LFTS** or **DNLS** (or noise). Each run writes a directory under
`../results/clustering/<run_name>/` containing `cluster_labels.csv` plus
diagnostic plots.

## Methods

| Method           | When to use                                                  |
|------------------|--------------------------------------------------------------|
| `dbscan`         | Density-based; identifies noise + clusters.                  |
| `kmeans`         | Forces exactly *N* clusters; no noise label.                 |
| `gmm`            | Gaussian Mixture; closest match to the Tanaka two-state model. |
| `dbscan_gmm`     | Production: DBSCAN denoising → GMM. Used in the paper.       |
| `hdbscan`        | Adaptive density; no `eps` to tune.                          |
| `hdbscan_gmm`    | HDBSCAN denoising → GMM.                                     |

Optional `--umap` reduces features to a low-D embedding before clustering.

## Active files

| File | Purpose |
|------|---------|
| `water_clustering.py`       | Main entry point for clustering methods and diagnostic plots. |
| `plot_style.py`             | Shared matplotlib style and feature labels. |
| `make_confidence_labels.py` | Convert clustering labels into confidence-cleaved paper labels. |
| `SFVS_metric.md`            | Archived specification for the old Structure-Factor Validation Score. |

Deprecated sweeps, replotters, standalone SFVS code, and one-off paper plots
were moved to `_archive/deprecated_20260716/`.

## Usage

DBSCAN→GMM (production):

```bash
python water_clustering.py \
    --mat_file  ../data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta_file ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --n_runs 1 \
    --method dbscan_gmm \
    --eps 0.05 --min_samples 30 \
    --features zeta_all \
    --out_dir ../results/clustering/tip4p2005_T-20_dbscan_gmm
```


→ Next: [Stage 4 — Structure Factor](../4_structure_factor/README.md)
