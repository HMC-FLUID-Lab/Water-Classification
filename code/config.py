"""
Shared paths and analysis settings for every stage of the pipeline.

Layout (relative to the deposit root):
    data/<FORCE FIELD>/trajectories/            MD trajectories (DCD + PDB + OpenMM system XML)
    data/<FORCE FIELD>/order_parameters/        per-molecule order parameters (.npz)
    data/<FORCE FIELD>/classification_labels/   per-molecule classification labels (.csv)
    results/                                     intermediate results (classifier, S(k), fractions)
    figures/                                     every generated figure panel
    tables/                                      every generated Supporting-Information table
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("WATER_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("WATER_RESULTS", os.path.join(ROOT, "results"))
FIGURES = os.environ.get("WATER_FIGURES", os.path.join(ROOT, "figures"))
TABLES = os.environ.get("WATER_TABLES", os.path.join(ROOT, "tables"))

# ── simulated conditions ─────────────────────────────────────────────────────
TEMPS = list(range(-30, 91, 10))                     # degrees Celsius
MODELS = {                                           # force field: independent runs
    "TIP4P-2005": ["Run01", "Run02", "Run03"],
    "TIP5P": ["Run01"],
    "SWM4-NDP": ["Run02"],
}
# The run whose first 20 frames represent each condition in Figs. 1, 3, 4b,c, S6, S7
MAIN_RUN = {"TIP4P-2005": "Run01", "TIP5P": "Run01", "SWM4-NDP": "Run02"}
N_MOL = 1024
N_FRAMES = 20                                         # frames per analyzed sample
N_FRAMES_RUN = 100                                    # TIP4P/2005 frames per run (Fig. 4a, Table S6)

# ── likelihood-ratio classifier ──────────────────────────────────────────────
FEATURES = ["zeta", "q", "LSI", "s_k"]               # the four order parameters (fit order)
FEATURES_5 = ["q", "Q6", "LSI", "s_k", "zeta"]       # benchmark pipelines (Table S4, Figs. S1-S5)
REFERENCE = ("TIP4P-2005", -20, "Run01")             # classifier is fitted here, then frozen
TRIM_EPS, TRIM_MIN_SAMPLES = 0.06, 20                 # density trim before the mixture fit
TAU = 2.0                                            # |Lambda| < tau -> transitional
SEED = 42
LABEL_NAMES = {0: "LFTS", 1: "DNLS", -1: "transitional"}

# ── structure factor ─────────────────────────────────────────────────────────
RC = 1.5                                             # nm, real-space window cutoff
R_OO = 0.285                                         # nm, reduced wavenumber k r_OO / 2 pi
K_VALUES_NM = (0.1, 50.0, 300)                       # S_g(k) grid (linspace arguments)
K_VALUES_ZETA_NM = (0.1, 50.0, 500)                  # S(k, zeta) grid
ZETA_BINS_A = (-1.0, 1.5, 41)                        # S(k, zeta) bin edges, angstrom


def tag(model, temp, run):
    return f"{model}_T{temp}C_{run}"


def trajectory(model, temp, run):
    base = os.path.join(DATA, model, "trajectories", tag(model, temp, run))
    return base + ".dcd", base + ".pdb"


def order_parameter_file(model, temp, run):
    return os.path.join(DATA, model, "order_parameters",
                        tag(model, temp, run) + "_order_parameters.npz")


def label_file(model, temp, run):
    return os.path.join(DATA, model, "classification_labels",
                        tag(model, temp, run) + "_labels.csv")


def results(*parts):
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
