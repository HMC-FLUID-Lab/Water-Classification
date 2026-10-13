# Data and code: *Is Water a Mixture: Unsupervised Likelihood-Ratio Classification Reveals Two Water States That Merge Upon Increasing Temperature*

Yiqi Yao, Diya Sanghi, Akanksha D. Chokshi, Sarah Nathani, Marcellinus Jerricho,
Minwoo Choi, Haiqin Wang, Bilin Zhuang (Harvey Mudd College; Yale-NUS College).
Correspondence: bzhuang@g.hmc.edu

This deposit contains the molecular-dynamics trajectories, the per-molecule order
parameters and classification labels, and the analysis code that produces every
data figure panel and every Supporting-Information (SI) table of the paper.

```
Water-Classification/   (branch doi-deposit)
├── README.md            this file
├── LICENSE              MIT (code); data are released under CC BY 4.0
├── requirements.txt     Python packages and versions
├── run_all.sh           rebuilds results/, figures/ and tables/ from data/
├── code/
│   ├── config.py                 paths and analysis settings shared by all stages
│   ├── 1_simulation/             OpenMM molecular dynamics
│   ├── 2_order_parameters/       q, LSI, s_k, zeta (and Q6) per molecule
│   ├── 3_classification/         likelihood-ratio classifier, labels, temperature series
│   ├── 4_structure_factor/       group structure factors S_g(k) and S(k, zeta)
│   ├── 5_figures/                one script per paper figure
│   └── 6_tables/                 SI tables
├── data/                (from Zenodo, not in git)
│   ├── TIP4P-2005/   trajectories/  order_parameters/  classification_labels/
│   ├── TIP5P/        trajectories/  order_parameters/  classification_labels/
│   └── SWM4-NDP/     trajectories/  order_parameters/  classification_labels/
├── results/             intermediate results (classifier, S(k), population series)
├── figures/             every generated figure panel (PNG, 600 dpi)
└── tables/              every generated SI table (CSV)
```

**Where the pieces are.** The code, figure panels, tables and intermediate results
are this repository (branch `doi-deposit`, archived on Zenodo, DOI: *to be added*).
The data are a separate Zenodo record (DOI: *to be added*) with one archive per
force field: `data_TIP4P-2005.zip`, `data_TIP5P.zip`, `data_SWM4-NDP.zip`. Unzip
them in the repository root to create `data/`:

```bash
git clone -b doi-deposit https://github.com/HMC-FLUID-Lab/Water-Classification.git
cd Water-Classification
unzip /path/to/data_TIP4P-2005.zip && unzip /path/to/data_TIP5P.zip && unzip /path/to/data_SWM4-NDP.zip
```

## Quick start

```bash
pip install -r requirements.txt     # Python 3.11
JOBS=16 ./run_all.sh                # stages 3-6; minutes, or ~15-30 min if S(k) is recomputed; no GPU
```

`run_all.sh` reads `data/` and rewrites `data/*/classification_labels/`,
`results/`, `figures/` and `tables/`. Structure factors that already exist in
`results/structure_factors/` are reused; delete that folder to recompute them
(this is the slowest step).

## Data

### Simulated conditions

N = 1024 water molecules, NVT, OpenMM 8.2, 1 fs time step, annealed from 373.15 K,
10 ns equilibration, then one frame every 10 ps. Temperatures −30 to +90 °C in
10 °C steps (13 temperatures). Density from the tabulated experimental
density of water, held at its +10 °C value (0.9997 g/cm³) for T ≤ +10 °C.

| Folder | Model | Integrator | Runs deposited | Used in |
|---|---|---|---|---|
| `TIP4P-2005/` | rigid, non-polarizable, four-site | Langevin, 20 ps⁻¹ | Run01, Run02, Run03 | Figs. 1, 3, 4; SI Figs. S1–S6; SI Tables S2–S7 |
| `TIP5P/` | rigid, non-polarizable, five-site | Langevin, 20 ps⁻¹ | Run01 | SI Figs. S6, S7; SI Tables S7, S8 |
| `SWM4-NDP/` | polarizable, Drude oscillator | Drude Langevin, 20 ps⁻¹; Drude 1 K, 5 ps⁻¹ | Run02 | SI Figs. S6, S7; SI Tables S7, S8 |

Every file is named `<FORCE FIELD>_T<temperature>C_<run>`, for example
`TIP4P-2005_T-20C_Run01`.

### `trajectories/`

| File | Content |
|---|---|
| `*.dcd` | first 10 ns of production (1000 frames, 10 ps apart). Exception: TIP4P/2005 Run02 and Run03 at +40 to +90 °C contain 420 frames (4.2 ns). |
| `*.pdb` | configuration at the end of equilibration; use it as the topology for the DCD |
| `*_system.xml` | the serialized OpenMM `System` (force field, box, constraints) |

Atoms per molecule: TIP4P/2005 O, H1, H2, M (4); TIP5P O, H1, H2 and two lone-pair
sites (5); SWM4-NDP O, H1, H2, M and the Drude particle (5). Only oxygens enter
the order parameters and structure factors.

### `order_parameters/` (`*_order_parameters.npz`)

NumPy archives with arrays of shape `[n_frames, 1024]`: row = trajectory frame
(starting at frame 0), column = molecule (trajectory order). Distances are in nm.

| Array | Meaning |
|---|---|
| `q` | orientational tetrahedral order parameter (4 nearest O neighbours) |
| `LSI` | local structure index, nm² (neighbours within 0.37 nm) |
| `s_k` | translational tetrahedral order parameter (4 nearest O neighbours) |
| `zeta` | local translational order parameter ζ, nm (9 nearest O neighbours; H bond: r_OO < 0.35 nm and an O–H···O angle below 30°) |
| `Q6` | bond-orientational order (TIP4P/2005 and the −20 °C TIP5P/SWM4-NDP files; used only by the SI benchmark pipelines) |

Frames covered: TIP4P/2005 the first 100 frames of each run; TIP5P and SWM4-NDP the
first 20 frames. These are all the frames analyzed in the paper.

```python
import numpy as np
d = np.load("data/TIP4P-2005/order_parameters/TIP4P-2005_T-20C_Run01_order_parameters.npz")
d["zeta"].shape        # (100, 1024)
```

### `classification_labels/` (`*_labels.csv`)

One row per molecule and frame of the matching order-parameter file:

| Column | Meaning |
|---|---|
| `frame`, `molecule` | indices into the order-parameter arrays and the trajectory |
| `Lambda` | log-likelihood ratio Λ = log p_H − log p_L of the frozen classifier |
| `label` | `LFTS` (Λ ≥ 2), `DNLS` (Λ ≤ −2) or `transitional` (\|Λ\| < 2) |

`TIP4P-2005_T-20C_Run01_benchmark_labels.csv` holds, for the first 20 frames of
the reference run, the partitions of the SI benchmark pipelines (`kmeans`, `gmm`,
`hdbscan_gmm`) and of the two density cuts of SI Table S5 (`posterior_cut`,
`fisher_valley_cut`), coded `H`, `L` or `set_aside`.

## The classifier

Fitted once on the reference sample (TIP4P/2005, −20 °C, Run01, first 20 frames =
20,480 molecules) and then applied unchanged (frozen) to every other temperature,
run and force field (`code/3_classification/classifier.py`):

1. Scale (ζ, q, LSI, s_k) to [0, 1] with the reference minima and maxima.
2. Density trim: molecules that are not core points of a fixed-radius density
   criterion (`sklearn.cluster.DBSCAN`, eps = 0.06, min_samples = 20) are left out of
   the mixture fit only (about 5% of the reference); every molecule is classified.
3. Two-component full-covariance Gaussian mixture (n_init = 5, random_state = 42);
   the component with the larger mean ζ is H (LFTS).
4. Λ(x) = log p_H(x) − log p_L(x) (component densities, weights not included);
   LFTS if Λ ≥ τ, DNLS if Λ ≤ −τ, transitional otherwise, τ = 2.

The fitted parameters are stored in `results/classifier/frozen_classifier.json`
(SI Table S3).

## Code

All scripts are run from their own folder and read their paths from `code/config.py`.

| Stage | Command | Output |
|---|---|---|
| 1 | `python run_md.py TIP4P-2005 -20 Run01` (in `1_simulation/`, OpenMM environment, GPU) | trajectory files |
| 2 | `python compute_order_parameters.py TIP4P-2005 -20 Run01 100` | `order_parameters/*.npz` |
| 2 | `python fast_order_parameters.py TIP5P 40 Run01` (vectorized; q, LSI, s_k, ζ) | `order_parameters/*.npz` |
| 3a | `python make_labels.py` | classifier, `classification_labels/*.csv` |
| 3b | `python temperature_series.py` | `results/temperature_series/*.csv` |
| 4 | `python compute_structure_factors.py` | `results/structure_factors/*.npz` |
| 5 | `python main_fig1.py` … `python appendix_fig7.py` | `figures/*.png` |
| 6 | `python appendix_tables.py` | `tables/*.csv` |

Stage 1 needs OpenMM 8.2 (Python 3.12) and a GPU (about a day per 10 ns run);
`tip5p.xml` and `swm4ndp.xml` ship with OpenMM and `tip4p2005.xml` is in
`1_simulation/`. Stage 2 takes minutes to hours per trajectory, depending on the
number of frames. Neither is re-run by `run_all.sh`. All TIP4P/2005 files and the
−20 °C TIP5P and SWM4-NDP files were computed with `compute_order_parameters.py`;
the other TIP5P and SWM4-NDP temperatures were computed with
`fast_order_parameters.py`, which implements the same definitions (maximum deviation
below 1e-6 at −20 °C, identical labels).

## Figure panels (`figures/`)

`main_` = main text, `appendix_` = Supporting Information (appendix_figN = Figure SN).
Panel letters follow the paper. Panel letters and titles are not drawn in the images.

| File | Paper | Script (`code/5_figures/`) |
|---|---|---|
| `main_fig1_a` | Fig. 1a, joint (ζ, q) density, all molecules | `main_fig1.py` |
| `main_fig1_b` | Fig. 1b, LFTS / DNLS density contours, transitional in grey | `main_fig1.py` |
| `main_fig1_c` | Fig. 1c, f(q), f(LSI), f(s_k), f(ζ) per state | `main_fig1.py` |
| `main_fig1_d` | Fig. 1d, (q, ζ) scatter of the classified molecules | `main_fig1.py` |
| — | Fig. 2, workflow schematic (drawn by hand, no data) | — |
| `main_fig3_a` | Fig. 3a, S_g(k) of LFTS, DNLS and transitional groups | `main_fig3.py` |
| `main_fig3_b_LFTS`, `main_fig3_b_DNLS` | Fig. 3b, S(k, ζ) surfaces (left, right) | `main_fig3.py` |
| `main_fig3_c_LFTS`, `main_fig3_c_DNLS` | Fig. 3c, S(k, ζ) contour maps (left, right) | `main_fig3.py` |
| `main_fig4_a` | Fig. 4a, state fractions vs T (mean of 3 runs) | `main_fig4.py` |
| `main_fig4_b` | Fig. 4b, S_H(k) of LFTS, −30 to +90 °C | `main_fig4.py` |
| `main_fig4_c` | Fig. 4c, S_L(k) of DNLS, −30 to +90 °C | `main_fig4.py` |
| `appendix_fig1_a`, `_b` | Fig. S1, K-means: distributions (a), (q, ζ) scatter (b) | `appendix_fig1.py` |
| `appendix_fig2_a`, `_b` | Fig. S2, GMM: distributions (a), scatter (b) | `appendix_fig2.py` |
| `appendix_fig3_a`, `_b` | Fig. S3, HDBSCAN→GMM: distributions (a), scatter (b) | `appendix_fig3.py` |
| `appendix_fig4_a`, `_b`, `_c` | Fig. S4, K-means, GMM, HDBSCAN→GMM in the (ζ, q) plane | `appendix_fig4.py` |
| `appendix_fig5_a`, `_b`, `_c` | Fig. S5, BIC, AIC, silhouette vs number of components | `appendix_fig5.py` |
| `appendix_fig6_a`, `_b`, `_c` | Fig. S6, S_g(k) at −20 °C: TIP4P/2005, TIP5P, SWM4-NDP | `appendix_fig6.py` |
| `appendix_fig7_a` … `_d` | Fig. S7, SWM4-NDP LFTS, SWM4-NDP DNLS, TIP5P LFTS, TIP5P DNLS vs T | `appendix_fig7.py` |

SI Figures S1–S3 have no panel letters in the paper; here `_a` is the block of four
distributions and `_b` is the scatter.

## SI tables (`tables/`)

| File | SI table |
|---|---|
| `appendix_table2_order_parameters.csv` | S2, order-parameter ranges and Cohen's d |
| `appendix_table3_mixture_means.csv`, `_covariances.csv`, `_scaler_ranges.csv` | S3, the frozen Gaussian mixture |
| `appendix_table4_pipelines.csv` | S4, benchmark pipelines |
| `appendix_table5_density.csv` | S5, local density of the set-aside molecules |
| `appendix_table6_temperature.csv` | S6, frozen-classifier and constant-quality series |
| `appendix_table7_model_transfer.csv` | S7, three water models at −20 °C |
| `appendix_table8_model_temperature.csv` | S8, TIP5P and SWM4-NDP vs temperature |

Table S1 lists simulation settings only (see "Simulated conditions" above).

## Reproducibility

Every stochastic step has a fixed seed (`random_state = 42` for mixtures and
K-means, `random_state = 0` for silhouette subsamples and plotting subsamples).
With the versions in `requirements.txt` the deposited code reproduces the
classification labels, all group structure factors, the S(k, ζ) surfaces and the
population series of the paper exactly. Known differences:

* HDBSCAN→GMM (SI Table S4, Figs. S3–S4) depends on the scikit-learn HDBSCAN
  implementation: scikit-learn 1.8.0 sets aside 14.5% of molecules (silhouette
  0.255), against 15.2% (0.257) in the original run.
* The likelihood-ratio silhouette evaluates to 0.347 here (SI Table S4: 0.340), and
  Cohen's d of s_k to 0.985 (SI Table S2: 0.99).

Figures use Nimbus Sans if it is installed (Debian/Ubuntu package
`fonts-urw-base35`) and otherwise DejaVu Sans, which changes only the typeface.

## License

Code: MIT (`LICENSE`). Data (trajectories, order parameters, labels, figures,
tables): Creative Commons Attribution 4.0 International (CC BY 4.0).
