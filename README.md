# Water ML-Clustering Pipeline

Analysis code for identifying **Locally Favored Tetrahedral Structures (LFTS)**
and **Disordered Normal-Liquid Structures (DNLS)** in water MD simulations,
following the Shi & Tanaka two-state framework (*JACS* **142**, 2868 — 2020).

Molecules are described by four local order parameters (ζ, *q*, LSI, *S*ₖ) and
sorted into the two states by an unsupervised **likelihood-ratio (LR)
classifier**: DBSCAN core detection → two-component Gaussian mixture → a
symmetric likelihood-ratio reject band. Molecules that fall in the ambiguous
band between the states are labelled **transition**. This is the method used in
the current manuscript (Draft 6).

> **Scope of this repository.** This repo tracks the **stage 1–4 analysis code**
> (simulation drivers, order parameters, clustering, structure factor). The
> orchestration scripts, raw data, results, figure builders, and the manuscript
> are large and are kept **local, not tracked** — see
> [Local working tree](#local-working-tree) for where they live.

---

## The two-state classifier (paper method)

Per molecule, on the four scaled features **[ζ, q, LSI, Sₖ]** (Q6 deliberately
excluded):

1. **Cores** — DBSCAN (`eps = 0.06`, `min_samples = 20`) isolates the dense
   cores and drops sparse noise.
2. **Mixture** — a 2-component full-covariance Gaussian mixture is fit to the
   cores, giving per-state densities *p*₍LFTS₎ and *p*₍DNLS₎.
3. **Likelihood ratio** — Λ(x) = log *p*₍LFTS₎(x) − log *p*₍DNLS₎(x).
   Assign **LFTS** if Λ ≥ +τ, **DNLS** if Λ ≤ −τ, else **transition**
   (ambiguous), with **τ = 2.0**.

For the generality figure the classifier is **fit once** on tip4p2005 at −20 °C
and then **frozen** (stored scaler + GMM + τ) and applied unchanged to every
other temperature and water model, so each panel isolates a single variable. The
reference implementation is `5_paper_figures/run_generality_lr.py`
(`fit_frozen_lr`), kept locally (see below).

---

## Pipeline (run in order)

```
  1_simulate           MD trajectories                        → data/simulations/*
  2_order_params       DCD → q, Q6, LSI, Sk, ζ                → data/order_params/*.mat
  3_clustering         MAT → cluster cores + two-state labels → results/clustering/<run>/
  4_structure_factor   labels + DCD → per-cluster S(k)        → results/clustering/<run>/
  5_paper_figures      LR relabelling + manuscript figures    → 6_paper_writing/.../images/  (local)
```

Stages **1–4** are the tracked analysis code. Stage **5** (figure builders), the
orchestration wrappers, and the `data/`→`results/` I/O are part of the local
working tree, not this repo.

---

## Repository layout (tracked)

```
.
├── README.md                  ← this file
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── 1_simulate/                ← OpenMM drivers + simulation engine
├── 2_order_params/            ← DCD → MAT (ζ, q, LSI, Sk, Q6)
├── 3_clustering/              ← clustering + two-state (LR) labelling
│   ├── water_clustering.py    ← clustering methods + diagnostic plots
│   ├── make_confidence_labels.py  ← GMM-posterior two-state labels (variant)
│   └── SFVS_metric.md         ← archived validation-score specification
└── 4_structure_factor/        ← per-cluster S(k) and S(k, ζ) helpers
```

Each stage directory has its own `README.md`.

---

## Running the tracked stages

The stage scripts read inputs from `data/` and write outputs to `results/` (in
the local working tree — see below).

### Stage 1 — Simulate

```bash
python 1_simulate/runWater_tip4p2005.py
python 1_simulate/runWater_tip5p.py
# Drude-polarizable SWM4-NDP needs swm4ndp.xml in the run dir:
mkdir -p data/simulations/swm4ndp && cd data/simulations/swm4ndp
python ../../../1_simulate/runWater_swm4ndp_multitemp.py
```

### Stage 2 — Order parameters

```bash
# One condition
python 2_order_params/run_single_condition.py tip4p2005 T-20 Run01
# Every DCD for a model
python 2_order_params/compute_order_params.py --model tip4p2005
```

### Stage 3 — Clustering + two-state labels

Base DBSCAN→GMM clustering (the LR reject band is layered on top; see the method
above and `run_generality_lr.py` in the local figure tree):

```bash
python 3_clustering/water_clustering.py \
    --mat_file  data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta_file data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --method dbscan_gmm --eps 0.06 --min_samples 20 \
    --features zeta_all \
    --out_dir results/clustering/tip4p2005_T-20_dbscan_gmm
```

### Stage 4 — Per-cluster structure factor

```bash
python 4_structure_factor/structure_factor_bycluster.py \
    --dcd-file       data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file       data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --zeta-file      data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --cluster-labels results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels_matrix_dbscan_gmm.csv \
    --cluster-only --model-name tip4p2005 --temperature -20 \
    --output-dir results/structure_factor/tip4p2005_T-20_dbscan_gmm
```

---

## Local working tree

The full runnable setup is kept **local and gitignored** (large binaries,
figures, and the manuscript). In this working tree:

```
7_file_collection/          ← local data + orchestration + outputs (gitignored)
├── pipeline/               ← run_pipeline.sh, auto_cluster_pipeline.py, batch drivers
├── data/                   ← inputs:  simulations/{tip4p2005,tip5p,swm4ndp}/, order_params/
├── results/                ← outputs: clustering/, structure_factor/, paper_figures/
└── (reference PDFs, figure backups)

5_paper_figures/            ← LR figure builders (gitignored)
│   run_generality_lr.py       fits the frozen LR classifier + writes LR caches
│   make_needed_photos.py      main-text figures (clustering, validation, generality)
│   make_ver2_skzeta{2,3}d.py  S(k, ζ) 2-D heatmaps + 3-D surfaces (Fig. 2)
│   make_ver2_figures.py       per-cluster S(k) FSDP comparison
│   make_si_figures.py         SI figures (cluster number, heatmaps, method comparison)

6_paper_writing/            ← manuscript (gitignored)
│   Paper_WaterMLClustering.zip            self-contained bundle (tex + images)
│   Paper_WaterMLClustering/Draft6.tex     current draft (LR method)
```

To reproduce the paper figures locally (from existing order params), fit the LR
classifier then render:

```bash
cd 5_paper_figures
python run_generality_lr.py all      # LR labels + S(k)/generality caches
python make_needed_photos.py         # main-text figures
python make_ver2_skzeta3d.py && python make_ver2_skzeta2d.py   # S(k,ζ) panels
python make_si_figures.py            # SI figures
```

### The manuscript

`6_paper_writing/Paper_WaterMLClustering.zip` (~22 MB) is the self-contained
bundle (all `.tex` + `images/` + previous drafts). To copy it to your local
machine, run this **from your own local terminal**:

```bash
scp water@134.173.35.25:/home/water/WaterSimulation/WaterClassification/6_paper_writing/Paper_WaterMLClustering.zip .
```

Build the PDF locally with `latexmk -pdf Draft6.tex` (LaTeX build artifacts are
gitignored).

---

## Dependencies

Python 3.11+. Install:

```
numpy scipy pandas scikit-learn matplotlib seaborn tqdm joblib
mdtraj openmm
umap-learn  # optional — only used by --umap flag in 3_clustering
```

`pip install -r requirements.txt` if you want to use the included list.

---

## License

Released under the [MIT License](LICENSE) — see the `LICENSE` file for full text.
