"""figstyle_pilot.py -- shared style and artifact paths for this leg's
figures, mirroring the parent repo's `figures/figstyle.py` (same principle:
every make_fig*.py here reads COMMITTED ARTIFACTS -- master_table.csv, flip
reports, train.jsonl logs -- not a re-run of the model, so figures
regenerate in seconds without a GPU or checkpoints present).

Palette values for `base`/`hidden`/`osim`/`peri`/`fexo` are copied verbatim
from the parent's `figures/figstyle.py` so a reader flipping between the
main paper's figures and this pilot's sees the same guide/reward colour
consistently. `nitrogen` and the db/rtb objective split are new -- this
pilot has axes the parent figures don't (a dense nitrogen-fraction control,
and only two objectives in scope, not four).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(*parts):
    return os.path.join(ROOT, *parts)


MASTER_TABLE = rel("results", "g2pt", "_aggregate", "master_table.csv")
FLIPS_DIR = rel("results", "g2pt", "flips")
LOGS_DIR = rel("logs", "g2pt-gfn")
OUT_DIR = rel("figures", "out")

# Copied verbatim from ../../figures/figstyle.py's GUIDE_COLOURS/BENCH_COLOURS
GUIDE_COLOURS = {"base": "#8C8C8C", "hidden": "#4C72B0"}
REWARD_COLOURS = {"osim": "#4C72B0", "peri": "#DD8452", "fexo": "#55A868",
                   "nitrogen": "#8172B2"}
OBJECTIVE_COLOURS = {"db": "#55A868", "rtb": "#C44E52"}

REWARD_TITLE = {"osim": "Osimertinib MPO", "peri": "Perindopril MPO",
                 "fexo": "Fexofenadine MPO", "nitrogen": "Nitrogen fraction"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def require(path, produced_by):
    if not os.path.exists(path):
        raise SystemExit(
            f"[FATAL] {path} does not exist yet.\n"
            f"        Produce it first: {produced_by}")


# Published GuacaMol baselines (Brown et al. 2019) and the shared GEOM-Drugs
# best-of-10k reference -- copied/pointed at the same source as
# ../../comparison_figures/figstyle_compare.py's PUBLISHED/GEOM_BASELINE_PATH,
# added 2026-09-05 once that reference existed (this file's docstring
# previously said "no published external baselines to draw"; that was true
# only until the GEOM-Drugs corpus and its reference score were shared
# across all three legs).
PUBLISHED = {
    "osim": {"REINVENT SMILES": 0.837, "ChEMBL best-of-dataset": 0.839},
    "peri": {"REINVENT SMILES": 0.537, "ChEMBL best-of-dataset": 0.575},
    "fexo": {"REINVENT SMILES": 0.784, "ChEMBL best-of-dataset": 0.817},
}
GEOM_BASELINE_PATH = rel("..", "..", "comparison_figures", "geom_baseline_best_of_10k.json")
