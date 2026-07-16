# Results (gitignored)

Generated outputs from stages 3, 4, 5. Layout:

```
results/
├── clustering/                          ← Stage 3/4 outputs
│   └── <run_name>/                      ← one per (model, temp, method)
│       ├── cluster_labels.csv           ← flat CSV: rows = molecules
│       ├── cluster_labels_matrix_*.csv   ← frames × molecules label matrices
│       ├── structure_factor_*.png        ← per-cluster S(k) plots
│       ├── *_scatter.png
│       ├── *_pairplot.png
│       └── *_zeta_dist.png
│
├── structure_factor/                    ← optional direct Stage 4 outputs
│
└── paper_figures/                       ← optional exported paper figures
```


Everything in this tree is regenerable by re-running the pipeline.
