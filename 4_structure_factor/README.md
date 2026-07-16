# Stage 4 — Structure Factor

Compute the Debye structure factor $S(k)$ per cluster (and across all atoms)
to verify that the clusters from Stage 3 correspond to physically distinct
ordered/disordered populations.

The active workflow focuses on **per-cluster S(k)** driven by Stage 3 cluster
labels. Older all-atoms/Tanaka reference scripts and plotting wrappers were
moved to `_archive/deprecated_20260716/`.

## Active files

| File | Purpose |
|------|---------|
| `structure_factor_bycluster.py` | Main entry: per-cluster S(k) from DCD + label matrix. |
| `sk_zeta_3d.py`                 | 3D S(k, ζ) surfaces used by the paper figure workflow. |

## Usage

Per-cluster S(k):

```bash
python structure_factor_bycluster.py \
    --dcd-file       ../data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file       ../data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --zeta-file      ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --cluster-labels ../results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels_matrix_dbscan_gmm.csv \
    --cluster-only \
    --model-name tip4p2005 --temperature -20 \
    --output-dir ../results/structure_factor/tip4p2005_T-20_dbscan_gmm
```

→ Next: [Stage 5 — Paper Figures](../5_paper_figures/README.md)
