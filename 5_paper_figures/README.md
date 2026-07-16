# 5_paper_figures

Code that generates the manuscript images in
`6_paper_writing/Paper_WaterMLClustering/images/`. After cleanup this directory
holds five modules; everything else was moved to `_archive/` (superseded
variants and experiments, kept for reference, not deleted).

## Files

| file | what it does |
|------|--------------|
| `nature_style.py` | Nature-Physics matplotlib style + colour system. Imported by everything. |
| `figlib.py` | Shared data loaders (`load_flat`, `load_cache`, `k_norm`) plus the reusable panel builders `fig1_two_states` (Fig. 1 top) and `build` (order-parameter distributions + scatter). |
| `build_data.py` | Builds the cached inputs from the MD trajectories + `.mat` order params. Run once. |
| `make_main_figures.py` | Main-text figures: fig1 (clustering), fig2 (validation), fig3 (generality). |
| `make_si_figures.py` | SI figures: cluster-number selection, DBSCAN/HDBSCAN hyperparameter heatmaps, method comparison, per-method order-parameter figures. |

## Which module makes which paper image

Main (`images/main/`):
- `1_clustering.png` — `make_main_figures.py fig1` → `figures_redesign/1_clustering_conf.png`
- `2_validation.png` — `make_main_figures.py fig2` (row a = `fig_validation_sk_conf.png`; rows b/c = `fig_skzeta_restyled_conf.png`, hand-assembled into the published composite)
- `3_generality.png` — `make_main_figures.py fig3` → `figures_redesign/figR4_generality.png`
- `0_flowchart.png` — hand-drawn, no script.

SI (`images/SI/`):
- `cluster_number.png` — `make_si_figures.py cluster_number`
- `dbscan_heatmap.png` / `hdbscan_heatmap.png` — `make_si_figures.py dbscan_heatmap hdbscan_heatmap`
- `method_comparison.png` — `make_si_figures.py method_comp`
- `op_kmeans.png` / `op_gmm.png` / `op_hdbscan.png` — `make_si_figures.py op_figures`

## Usage

```bash
# 1. build caches once (needs MD trajectories + order-param .mat files)
python build_data.py            # conf_sk, redesign, blocktrim

# 2. render figures from the caches (fast)
python make_main_figures.py     # fig1 fig2 fig3
python make_si_figures.py       # all SI figures
```

Both figure scripts accept figure names (`python make_main_figures.py fig3`) and
default to `all`. `make_si_figures.py --recompute` rebuilds the DBSCAN/HDBSCAN
grids instead of using the cached `heatmap_redesign_preview/grid_*.npz`.

Note: the published `2_validation.png` was assembled by hand from row a and the
S(k,ζ) panels; `make_main_figures.py` regenerates those panels but
`assemble_fig2()` only stitches them if the pre-cropped strips
(`fig_skzeta_conf_rowb/rowc.png`) are present.

Cross-directory dependencies (kept, not archived): `3_clustering/`
(`water_clustering.py`, `make_confidence_labels.py`, `plot_style.py`) and
`4_structure_factor/` (`structure_factor_bycluster.py`, `sk_zeta_3d.py`).
