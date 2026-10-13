#!/usr/bin/env python3
"""
Stage 1: one NVT molecular-dynamics run of N = 1024 water molecules (OpenMM).

    python run_md.py <force field> <temperature in C> <run>
    python run_md.py TIP4P-2005 -20 Run01
    python run_md.py TIP5P      40  Run01
    python run_md.py SWM4-NDP   0   Run02

Settings (main-text Computational Methods, SI Table S1): 1 fs time step, Langevin
integrator with a 20 ps^-1 friction (SWM4-NDP: Drude Langevin integrator, Drude
particles at 1 K with 5 ps^-1), annealing from 373.15 K, 10 ns equilibration,
then frames every 10 ps. The box volume follows the tabulated experimental
density of water (MDWater.Water_v), held at its +10 C value below +10 C.

The files are written to the working directory with the original naming,
dcd_<model>_T<T>_N1024_<run>_<i>.dcd (1000 frames = 10 ns per file),
inistate_<...>.pdb (configuration after equilibration, used as topology) and
system_<...>.xml; data/<FORCE FIELD>/trajectories/ holds the first production
file of each run, renamed to <FORCE FIELD>_T<T>C_<run>.dcd/.pdb/_system.xml.

Environment: OpenMM 8.2 (Python 3.12). tip5p.xml and swm4ndp.xml ship with
OpenMM; TIP4P/2005 uses tip4p2005.xml in this folder. GPU recommended
(WATER_PLATFORM=CUDA or OpenCL; default OpenCL).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from MDWater import MDWater, nanoseconds, picoseconds  # noqa: E402

FORCE_FIELD = {"TIP4P-2005": "tip4p2005", "TIP5P": "tip5p", "SWM4-NDP": "swm4ndp"}
FF_XML = {"tip4p2005": os.path.join(_HERE, "tip4p2005.xml")}


def run(force_field, temp_c, run_id, production_ns=10.0):
    model = FORCE_FIELD[force_field]
    MDWater(
        RunName=f"{model}_T{int(temp_c)}_N1024_{run_id}",
        Nwater=1024,
        T=float(temp_c),
        water_forcefield=model,
        t_equilibrate=10 * nanoseconds,
        t_simulate=production_ns * nanoseconds,
        t_reportinterval=10 * picoseconds,
        t_step=0.001 * picoseconds,
        CheckPointFileAvail=False,
        InitPositionPDB=None,
        ReportVelocity=False,
        ForceFieldChoice=FF_XML.get(model),
        PlatformName=os.environ.get("WATER_PLATFORM", "OpenCL"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in FORCE_FIELD:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], float(sys.argv[2]), sys.argv[3],
        float(os.environ.get("WATER_PRODUCTION_NS", 10)))
