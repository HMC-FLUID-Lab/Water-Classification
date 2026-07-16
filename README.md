# Water ML-Clustering Pipeline

End-to-end pipeline for identifying **Locally Favored Tetrahedral Structures
(LFTS)** and **Disordered Normal-Liquid Structures (DNLS)** in water MD
simulations, following the Shi & Tanaka two-state framework
(*JACS* **142**, 2868 — 2020).

The custom Structure-Factor Validation Score is documented in
[`3_clustering/SFVS_metric.md`](3_clustering/SFVS_metric.md). The active
workflow keeps the metric description while the older standalone score
implementation is archived.

---

## Pipeline (run in order)

```
  1_simulate           MD trajectories                        → data/simulations/*
  2_order_params       DCD → q, Q6, LSI, Sk, ζ                → data/order_params/*.mat
  3_clustering         MAT → cluster labels                   → results/clustering/<run>/
  4_structure_factor   labels + DCD → per-cluster S(k)        → results/clustering/<run>/
  5_paper_figures      composite figures used in the paper    → 5_paper_figures/figures_redesign/
```

The numbered prefix is the running order: stage *N* consumes the output of
stage *N − 1*. Each directory has its own `README.md` documenting the stage.

---

## Quickstart

Single condition, all five stages:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01
```

Skip the (expensive) MD step and start from existing DCDs:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --skip 2
```

Run only one stage:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --only 3
```

Batch every (model × temperature) at once:

```bash
bash pipeline/run_batch.sh        # clustering + S(k) sweep
bash pipeline/run_sk_batch.sh     # S(k) only, post-process existing batches
```

---

## Repository layout

```
.
├── README.md                  ← this file
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── 1_simulate/                ← OpenMM drivers + simulation engine
├── 2_order_params/            ← DCD → MAT (LSI, q, Sk, Q6, ζ)
├── 3_clustering/              ← active clustering entry point + style helpers
│   └── SFVS_metric.md         ← archived validation-score specification
├── 4_structure_factor/        ← per-cluster S(k) and S(k, ζ) helpers
├── 5_paper_figures/           ← manuscript figure builders
│
├── pipeline/                  ← orchestration scripts
│   ├── run_pipeline.sh        ← top-level driver (stages 1 → 5)
│   ├── auto_cluster_pipeline.py
│   ├── run_sk_from_batch.py
│   ├── run_batch.sh           ← batch clustering + S(k)
│   └── run_sk_batch.sh        ← batch S(k) post-process
│
├── data/                      ← inputs (gitignored — large binaries)
│   ├── simulations/{tip4p2005, tip5p, swm4ndp}/
│   └── order_params/
│
├── results/                   ← outputs (gitignored)
│   ├── clustering/
│   ├── structure_factor/
│   └── paper_figures/
│
└── _archive/                  ← legacy code kept locally for reference (gitignored)
```

**Code is committed; data and results are not.** See [.gitignore](.gitignore).

---

## Stage cheat sheet

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

### Stage 3/4 — Clustering and per-cluster structure factor

```bash
python pipeline/auto_cluster_pipeline.py \
    --mat-file  data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta-file data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --dcd-file  data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file  data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --method dbscan_gmm --eps 0.05 --min-samples 30 \
    --output-dir results/clustering/tip4p2005_T-20_dbscan_gmm
```

To run clustering only and skip S(k):

```bash
python pipeline/auto_cluster_pipeline.py ... --skip-structure-factor
```

`3_clustering/water_clustering.py` remains available when only label generation
and diagnostic clustering plots are needed.

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

### Stage 5 — Paper figures

```bash
python 5_paper_figures/build_data.py       # rebuild caches when inputs change
python 5_paper_figures/make_main_figures.py
python 5_paper_figures/make_si_figures.py
```

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
