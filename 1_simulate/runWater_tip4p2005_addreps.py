#!/usr/bin/env python3
"""
Add extra independent replicas of the TIP4P/2005 temperature sweep.

The original driver (runWater_tip4p2005.py) produced Run01 + Run02 for each of
the 9 temperatures (-40, -35, -30, -20, -10, 0, 10, 20, 30 degC). This script
launches ADDITIONAL replicas (Run03 by default) using identical production
settings, so populations / structure factors gain a third independent sample
for error bars. Each replica is independent: initial positions use unseeded
`random.randrange` and OpenMM assigns a unique Langevin seed per run.

SAFETY: a run is SKIPPED if its first production file (dcd_<name>_0.dcd) already
exists in the working directory, so this can never overwrite Run01/Run02 and is
safe to re-run / resume.

Run FROM the output directory so files land next to the existing runs:

    cd /home/water/WaterSimulation/WaterClassification/data/simulations/tip4p2005
    python /home/water/WaterSimulation/WaterClassification/1_simulate/runWater_tip4p2005_addreps.py

Single run (useful for one-GPU boxes / manual scheduling):

    python .../runWater_tip4p2005_addreps.py one tip4p2005_T-40_N1024_Run03 -40

Parallelism (GPU note below) is controlled by the WATER_NJOBS env var:

    WATER_NJOBS=1 python .../runWater_tip4p2005_addreps.py   # sequential (safest for 1 GPU)
"""
import os
import sys

# Make MDWater + its siblings importable no matter what the working directory is.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from MDWater import *  # noqa: F401,F403  (pulls in openmm units, MDWater, etc.)


# --- What to run -----------------------------------------------------------
# Same 9-point grid as runWater_tip4p2005.py (5 degC spacing in the deep
# supercooled region near the Schottky temperature, 10 degC elsewhere).
TEMPERATURES = [-40.0, -35.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

# Replicas to ADD. Run01/Run02 already exist -> add Run03 for n=3 total.
# For n=4, make this ["Run03", "Run04"].
REPLICAS = ["Run03"]

inputs_list = [
    (f"tip4p2005_T{int(T)}_N1024_{rep}", T)
    for rep in REPLICAS
    for T in TEMPERATURES
]


### The function to be run (identical production settings to runWater_tip4p2005.py) ###
def RunMD(inputs):
    run_name, T = inputs

    # Skip if this replica already has production output in the CWD.
    sentinel = f"dcd_{run_name}_0.dcd"
    if os.path.exists(sentinel):
        print(f"[skip] {run_name}: {sentinel} already exists")
        return

    MDWater(
        RunName=run_name,
        Nwater=1024,
        T=T,
        water_forcefield="tip4p2005",     # TIP4P/2005 model from Tanaka paper
        t_equilibrate=10 * nanoseconds,   # match production equilibration
        t_simulate=50 * nanoseconds,      # 50 ns production
        t_reportinterval=10 * picoseconds,
        t_step=0.001 * picoseconds,       # 1 fs timestep
        CheckPointFileAvail=False,
        InitPositionPDB=None,
        ReportVelocity=False,
        # This OpenMM ships no top-level tip4p2005.xml; point at our regenerated
        # copy (exact match to the original Run01/Run02 systems) by absolute path.
        ForceFieldChoice=os.path.join(_HERE, "tip4p2005.xml"),
        # Platform via env (WATER_PLATFORM); default OpenCL. Use CUDA + per-batch
        # CUDA_VISIBLE_DEVICES to pin a run to a specific GPU.
        PlatformName=os.environ.get("WATER_PLATFORM", "OpenCL"),
    )


if __name__ == "__main__":
    # Single-run mode: python ... one <run_name> <T_celsius>
    if len(sys.argv) >= 4 and sys.argv[1] == "one":
        RunMD((sys.argv[2], float(sys.argv[3])))
        sys.exit(0)

    from multiprocessing import cpu_count
    from joblib import Parallel, delayed

    n_jobs = int(os.environ.get("WATER_NJOBS", cpu_count()))
    print(f"TIP4P/2005 add-replicas | n_jobs={n_jobs} | runs={len(inputs_list)}")
    for name, T in inputs_list:
        print(f"   {name}  (T={T:g} degC)")
    Parallel(n_jobs=n_jobs)(delayed(RunMD)(i) for i in inputs_list)
