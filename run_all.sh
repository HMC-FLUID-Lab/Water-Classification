#!/usr/bin/env bash
# Rebuild every result, figure panel and table from the deposited order parameters
# and trajectories (stages 3-6). Stages 1-2 (MD and order parameters) are not re-run;
# see README.md.  Runtime: a few minutes when results/structure_factors/ exists,
# about 15 min on 40 CPU cores when it is recomputed (JOBS=40); no GPU needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
JOBS="${JOBS:-16}"

step() { echo; echo "=================== $* ==================="; }

cd "$HERE/code/3_classification"
step "3a  fit + freeze the classifier, label every molecule"
python make_labels.py
step "3b  temperature series (Fig. 4a, Table S6)"
python temperature_series.py

cd "$HERE/code/4_structure_factor"
step "4   group structure factors S_g(k) and S(k, zeta)"
python compute_structure_factors.py --jobs "$JOBS"

cd "$HERE/code/5_figures"
step "5   figure panels -> figures/"
for f in main_fig1 main_fig3 main_fig4 appendix_fig1 appendix_fig2 appendix_fig3 \
         appendix_fig4 appendix_fig5 appendix_fig6 appendix_fig7; do
  python "$f.py"
done

cd "$HERE/code/6_tables"
step "6   Supporting-Information tables -> tables/"
python appendix_tables.py

echo; echo "done."
