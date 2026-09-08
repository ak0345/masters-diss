"""figstyle_compare.py -- shared style and artifact paths for the three-way
comparison figures (Quetzal / MolGPT / G2PT).

Every make_cmp*.py script here reads COMMITTED ARTIFACTS from all three legs
(each leg's own master_table.csv / flip reports / dumps), never re-running
any model. Palette is per-MODEL (not per-guide or per-reward, since that's
this directory's whole axis of comparison) -- kept visually distinct from
each leg's own figstyle_pilot.py, which colours by guide/reward instead.
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(*parts):
    return os.path.join(ROOT, *parts)


OUT_DIR = rel("comparison_figures", "out")

MODEL_COLOURS = {"Quetzal": "#333333", "MolGPT": "#4C72B0", "G2PT": "#C44E52"}
MODEL_MARKERS = {"Quetzal": "s", "MolGPT": "o", "G2PT": "^"}

MASTER_TABLES = {
    "Quetzal": rel("results", "dumps", "_aggregate", "master_table.csv"),
    "MolGPT": rel("molgpt", "results", "molgpt", "_aggregate", "master_table.csv"),
    "G2PT": rel("g2pt", "results", "g2pt", "_aggregate", "master_table.csv"),
}
FLIPS_DIRS = {
    "Quetzal": rel("results", "flips-guide"),
    "MolGPT": rel("molgpt", "results", "molgpt", "flips"),
    "G2PT": rel("g2pt", "results", "g2pt", "flips"),
}
DUMPS_DIRS = {
    "Quetzal": rel("results", "dumps"),
    "MolGPT": rel("molgpt", "results", "molgpt", "dumps"),
    "G2PT": rel("g2pt", "results", "g2pt", "dumps"),
}
GEOM_REF = rel("shared_data", "geom_drugs_filtered.smi")

# Rewards common to all three legs' pilot grids (Quetzal's own sweep also has
# "zaleplon", not shared by the pilots, so it's excluded from every comparison
# here -- see decisions.md, "same dataset/vocab" entries, for why the pilots'
# grid was deliberately kept to this subset).
COMMON_REWARDS = ("nitrogen", "osim", "peri", "fexo")
REWARD_TITLE = {"osim": "Osimertinib MPO", "peri": "Perindopril MPO",
                 "fexo": "Fexofenadine MPO", "nitrogen": "Nitrogen fraction"}

# Published GuacaMol baselines, Brown et al. (2019) -- copied verbatim from
# ../figures/figstyle.py's PUBLISHED dict (no zaleplon/nitrogen row there
# either, same reason: not in that paper's Table 2).
PUBLISHED = {
    "osim": {"REINVENT SMILES": 0.837, "ChEMBL best-of-dataset": 0.839},
    "peri": {"REINVENT SMILES": 0.537, "ChEMBL best-of-dataset": 0.575},
    "fexo": {"REINVENT SMILES": 0.784, "ChEMBL best-of-dataset": 0.817},
}

# Best-of-10,000 score on a random sample of the *shared, filtered* GEOM-Drugs
# corpus (shared_data/geom_drugs_filtered.smi, seed 0) under each reward's own
# scoring function -- computed once by comparison_figures/compute_geom_baseline.py
# and cached here as a plain dict (not re-run per figure) since it's the same
# reference set for all three legs now (unlike Quetzal's own GEOM baseline,
# computed from its own 3D-native GEOM subset before the pilots existed).
GEOM_BASELINE_PATH = rel("comparison_figures", "geom_baseline_best_of_10k.json")

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
    if not path or not os.path.exists(path):
        raise SystemExit(
            f"[FATAL] {path} does not exist yet.\n"
            f"        Produce it first: {produced_by}")


def load_master(model):
    import csv
    path = MASTER_TABLES[model]
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def flip_reports(model):
    """label -> parsed flip_report json, for one model."""
    out = {}
    for f in sorted(glob.glob(f"{FLIPS_DIRS[model]}/flip_report_*.json")):
        d = json.load(open(f))
        out[d["label"]] = d
    return out


def t1_block(model, report, regime="atom_steps"):
    """Return the T=1.0 metrics dict for one flip report, from whichever of
    the THREE actually-different schemas this repo ended up with (verified
    by reading each, not assumed from one being a copy of another):
      - Quetzal:  report["flip_temp1.0"]                          (flat)
      - MolGPT:   report["results_by_temp"]["t1.0"]                (flat)
      - G2PT:     report["results_by_temp"]["t1.0"]["by_regime"][regime]
    G2PT's `state_count_by_position` key is also spelled
    `states_by_position` in Quetzal's own schema -- callers that need the
    per-position count list should use `state_count_key(model)` below
    rather than hardcoding a name.
    """
    if model == "Quetzal":
        return report.get("flip_temp1.0", {})
    t1 = (report.get("results_by_temp") or {}).get("t1.0", {})
    if model == "G2PT":
        return (t1.get("by_regime") or {}).get(regime, {})
    return t1


def state_count_key(model):
    return "states_by_position" if model == "Quetzal" else "state_count_by_position"


def flip_metric(report, model, metric, regime="atom_steps"):
    """Read one scalar flip metric (delivered_frac, argmax_flip_rate, ...)
    from a single model's report."""
    return t1_block(model, report, regime).get(metric)


NAME_RE = re.compile(
    r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-"
    r"replay_(?P<replay>on|off)-b(?P<beta>[0-9.]+)(?:-s(?P<seed>[0-9]+))?")
