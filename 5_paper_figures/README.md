# 5_paper_figures

Code that generates the manuscript images in
`6_paper_writing/Paper_WaterMLClustering/images/`. The current main-text figures
use the **likelihood-ratio (LR) two-state classifier** (see the top-level
README). Superseded variants and experiments were moved to `_archive/` (kept for
reference, not deleted).

## Files

| file | what it does |
|------|--------------|
| `nature_style.py` | Nature-Physics matplotlib style + colour system. Imported by everything. |
| `figlib.py` | Shared data loaders (`load_flat`, `load_cache`, `k_norm`) plus reusable panel builders. |
| `build_data.py` | Builds the cached inputs from the MD trajectories + `.mat` order params. Run once. |
| `run_generality_lr.py` | Fits the **frozen LR classifier** and writes LR labels + S(k)/generality caches for every condition. Run before the LR figure scripts. |
| `make_needed_photos.py` | Main-text figures (clustering, validation line, generality) from the LR caches. |
| `make_ver2_skzeta3d.py` / `make_ver2_skzeta2d.py` | LR S(k, ζ) 3-D surfaces + 2-D heatmaps (Fig. 2 panels). |
| `make_ver2_figures.py` | LR per-cluster S(k) with the FSDP pre-peak comparison. |
| `make_si_figures.py` | SI figures: cluster-number selection, DBSCAN/HDBSCAN hyperparameter heatmaps, method comparison, per-method order-parameter figures. |

The superseded confidence-method main renderer (`make_main_figures.py`) lives in
`_archive/` (gitignored).

## Which module makes which paper image

Main (`images/main/`):
- `1_classification.png` — `make_needed_photos.py` (clustering panels, LR)
- `2_result.png` — `make_ver2_skzeta3d.py` + `make_ver2_skzeta2d.py` + `make_ver2_figures.py`, hand-assembled into the published composite
- `3_generality.png` — `make_needed_photos.py` (S(k) vs T, population(T), model comparison)
- `0_flowchart.png` — hand-drawn, no script.

SI (`images/SI/`):
- `cluster_number.png` — `make_si_figures.py cluster_number`
- `dbscan_heatmap.png` / `hdbscan_heatmap.png` — `make_si_figures.py dbscan_heatmap hdbscan_heatmap`
- `method_comparison.png` — `make_si_figures.py method_comp`
- `op_kmeans.png` / `op_gmm.png` / `op_hdbscan.png` — `make_si_figures.py op_figures`

## Usage

```bash
# 1. build caches once (needs MD trajectories + order-param .mat files)
python build_data.py

# 2. fit the frozen LR classifier + write LR labels/caches
python run_generality_lr.py all

# 3. render figures from the caches (fast)
python make_needed_photos.py         # main-text clustering / validation / generality
python make_ver2_skzeta3d.py         # S(k,ζ) 3-D surfaces (Fig. 2)
python make_ver2_skzeta2d.py         # S(k,ζ) 2-D heatmaps (Fig. 2)
python make_ver2_figures.py          # per-cluster S(k) FSDP comparison
python make_si_figures.py            # all SI figures
```

`make_si_figures.py --recompute` rebuilds the DBSCAN/HDBSCAN grids instead of
using the cached `heatmap_redesign_preview/grid_*.npz`. The published
`2_result.png` was assembled by hand from the per-cluster S(k) row and the
S(k, ζ) panels.

Cross-directory dependencies (kept, not archived): `3_clustering/`
(`water_clustering.py`, `make_confidence_labels.py`, `plot_style.py`) and
`4_structure_factor/` (`structure_factor_bycluster.py`, `sk_zeta_3d.py`).
