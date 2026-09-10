"""make_cmp06_nn_similarity.py -- three-way comparison

    !!! KNOWN BIAS IN PANEL A -- READ BEFORE USING THIS FIGURE !!!

    `guided_vs_prior` compares each guided pool against the prior dump carrying the SAME
    sampling seed. Those two share a random stream, so 21-67% of their molecules are
    byte-for-byte identical, and an identical molecule scores a maximum Tanimoto of exactly
    1.0 against a reference containing it. `null_vs_prior` has no such coupling: it compares
    prior seed 42 against prior seed 0.

    The guided bar is therefore inflated relative to its own null by an amount set by how
    often the guide changes nothing, not by chemistry. Measured against an INDEPENDENT prior
    draw, Quetzal's guided pool returns 0.362-0.366 against a null of 0.366, i.e. no effect,
    where this script reports 0.507-0.569.

    An earlier reading of this figure concluded that Quetzal's guide concentrates its output.
    That conclusion was withdrawn on 2026-09-08. Use
    results/ablations/chemistry/analyse_chemistry.py, which uses an independent reference for
    both guided and null. This script is kept because the manuscript discusses the bias
    explicitly and needs the figure that exhibits it.
 of each leg's own
fig15 panel A/B headline statistic: does the guide concentrate samples
closer to chemistry the frozen prior already reaches (panel A), and does it
move away from or stay pinned to the shared GEOM-Drugs training corpus
(panel B)?

This is the direct, quantitative test of "the guide re-ranks within the
prior's existing distribution rather than extending it into new chemical
space" -- see ../figures/make_fig15_chemspace.py's own docstring, which
treats exactly this statistic (mean max-Tanimoto nearest-neighbour
similarity) as its headline result ("Panels A and B are therefore the
quantitative ones").

For each model: "prior" = one unguided draw from the frozen prior, "null" =
an independent second unguided draw of the identical frozen prior (the
baseline every guided curve is read against), "guided" = the per-reward
guided pool. Quetzal's prior/null are `_base/<reward>/seed{0,42}/
base_smiles.txt` (reward-specific); MolGPT's/G2PT's are base_smiles.txt
pooled across every sweep-*-s0 (prior) / sweep-*-s42 (null) config
(reward-agnostic -- the frozen prior's own unconditional generation does not
depend on which reward a config was later guided toward).

A point sitting to the RIGHT of (above) its own model's null, on panel A,
means that model's guide concentrates mass inside the prior's reachable set
rather than moving away from it -- the mechanism this whole comparison is
testing for. A point at or below null means the guide is statistically
indistinguishable from just drawing again from the unguided prior.
"""
import glob
import os
import random
import re
import sys

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, savefig, require, plt,
                               MODEL_COLOURS, MODEL_MARKERS, COMMON_REWARDS,
                               REWARD_TITLE, DUMPS_DIRS)

GEOM_SMI = rel("shared_data", "geom_drugs_filtered.smi")
N_GEOM = 1000
N_PER_CATEGORY = 300
SEED = 0

QUETZAL_NAME_RE = re.compile(
    r"^sweep-(?P<reward>[a-z]+)-(?P<guide>base|hidden)-(?P<objective>db|rtb)"
    r"-replay_(?P<replay>on|off)-b(?P<beta>\d+)(?:-s\d+)?$")


def fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_fps(smis, rng, n):
    if len(smis) > n:
        smis = rng.sample(smis, n)
    fps = [fingerprint(s) for s in smis]
    return [f for f in fps if f is not None]


def nn_to(query, ref):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in query], dtype=float)


def sample_lines(path, n, rng):
    if not os.path.exists(path):
        return []
    lines = [l.strip() for l in open(path) if l.strip()]
    return rng.sample(lines, n) if len(lines) > n else lines


def pooled(pattern, filename):
    out = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, filename)
        if os.path.exists(f):
            out += [l.strip() for l in open(f) if l.strip()]
    return out


def quetzal_pools():
    """(prior, null, {reward: guided}) -- prior/null are reward-specific for
    Quetzal (unlike the pilots), so this returns per-reward prior/null too."""
    prior, null, guided = {}, {}, {r: [] for r in COMMON_REWARDS}
    for r in COMMON_REWARDS:
        prior[r] = pooled(f"{rel('results','dumps','_base')}/{r}/seed0", "base_smiles.txt")
        null[r] = pooled(f"{rel('results','dumps','_base')}/{r}/seed42", "base_smiles.txt")
    for d in sorted(glob.glob(f"{DUMPS_DIRS['Quetzal']}/sweep-*")):
        m = QUETZAL_NAME_RE.match(os.path.basename(d))
        if not m or m.group("replay") != "off" or m.group("beta") != "10":
            continue
        reward = m.group("reward")
        if reward not in COMMON_REWARDS:
            continue
        f = os.path.join(d, "seed0", "guided_smiles.txt")
        if os.path.exists(f):
            guided[reward] += [l.strip() for l in open(f) if l.strip()]
    return prior, null, guided


def pilot_pools(model):
    """Same shape as quetzal_pools, but prior/null are reward-agnostic
    (a single pool reused for every reward) since MolGPT's/G2PT's frozen
    prior generation doesn't depend on which reward a config targets."""
    d = DUMPS_DIRS[model]
    prior_pool = pooled(f"{d}/sweep-*-s0", "base_smiles.txt")
    null_pool = pooled(f"{d}/sweep-*-s42", "base_smiles.txt")
    prior = {r: prior_pool for r in COMMON_REWARDS}
    null = {r: null_pool for r in COMMON_REWARDS}
    guided = {r: [] for r in COMMON_REWARDS}
    for dd in sorted(glob.glob(f"{d}/sweep-*")):
        reward = os.path.basename(dd).split("-")[1]
        if reward not in COMMON_REWARDS:
            continue
        f = os.path.join(dd, "guided_smiles.txt")
        if os.path.exists(f):
            guided[reward] += [l.strip() for l in open(f) if l.strip()]
    return prior, null, guided


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(DUMPS_DIRS[model], f"produce {model}'s dumps first")

    rng = random.Random(SEED)
    geom_fps = load_fps(sample_lines(GEOM_SMI, N_GEOM, rng), rng, N_GEOM)

    raw_pools = {"Quetzal": quetzal_pools(), "MolGPT": pilot_pools("MolGPT"),
                 "G2PT": pilot_pools("G2PT")}

    # {model: {reward: {"null_vs_prior":.., "guided_vs_prior":.., "prior_vs_geom":.., "guided_vs_geom":..}}}
    stats = {}
    for model, (prior, null, guided) in raw_pools.items():
        stats[model] = {}
        for r in COMMON_REWARDS:
            prior_fps = load_fps(prior[r], rng, N_PER_CATEGORY)
            null_fps = load_fps(null[r], rng, N_PER_CATEGORY)
            guided_fps = load_fps(guided[r], rng, N_PER_CATEGORY)
            if len(prior_fps) < 10 or len(null_fps) < 10 or len(guided_fps) < 10:
                print(f"[cmp06] {model}/{r}: too few valid molecules "
                      f"(prior={len(prior_fps)} null={len(null_fps)} guided={len(guided_fps)}), skipping")
                continue
            stats[model][r] = {
                "null_vs_prior": nn_to(null_fps, prior_fps).mean(),
                "guided_vs_prior": nn_to(guided_fps, prior_fps).mean(),
                "prior_vs_geom": nn_to(prior_fps, geom_fps).mean(),
                "guided_vs_geom": nn_to(guided_fps, geom_fps).mean(),
            }
            if model == "Quetzal":
                # The corrected reading: guided (seed 0) against the INDEPENDENT
                # prior draw (seed 42) that "null_vs_prior" already uses as its
                # own reference, rather than against the seed-matched prior
                # "guided_vs_prior" uses above. Only meaningful for Quetzal,
                # where prior/null are genuinely seed-paired with guided; the
                # pilots' "guided_vs_prior" is not seed-matched to begin with.
                stats[model][r]["guided_vs_null"] = nn_to(guided_fps, null_fps).mean()
            print(f"[cmp06] {model}/{r}: {stats[model][r]}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---------------- left: distance to prior (panel A statistic) ----------
    ax = axes[0]
    models = ["Quetzal", "MolGPT", "G2PT"]
    x = np.arange(len(COMMON_REWARDS))
    width = 0.25
    for i, model in enumerate(models):
        nulls = [stats[model].get(r, {}).get("null_vs_prior", np.nan) for r in COMMON_REWARDS]
        guides = [stats[model].get(r, {}).get("guided_vs_prior", np.nan) for r in COMMON_REWARDS]
        xi = x + (i - 1) * width
        ax.bar(xi, nulls, width * 0.9, color=MODEL_COLOURS[model], alpha=0.35, zorder=2)
        ax.scatter(xi, guides, color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
                   s=50, zorder=3, label=model, edgecolors="k", linewidths=0.5)
        if model == "Quetzal":
            indep = [stats[model].get(r, {}).get("guided_vs_null", np.nan) for r in COMMON_REWARDS]
            ax.scatter(xi, indep, facecolors="none", edgecolors=MODEL_COLOURS[model],
                      marker=MODEL_MARKERS[model], s=50, zorder=4, linewidths=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_TITLE.get(r, r) for r in COMMON_REWARDS], rotation=15, fontsize=8)
    ax.set_ylabel("mean max-Tanimoto similarity")
    ax.set_title("A. Against the null", fontsize=9)

    # ---------------- right: distance to GEOM (panel B statistic) ----------
    ax = axes[1]
    for i, model in enumerate(models):
        priors = [stats[model].get(r, {}).get("prior_vs_geom", np.nan) for r in COMMON_REWARDS]
        guides = [stats[model].get(r, {}).get("guided_vs_geom", np.nan) for r in COMMON_REWARDS]
        xi = x + (i - 1) * width
        ax.bar(xi, priors, width * 0.9, color=MODEL_COLOURS[model], alpha=0.35, zorder=2)
        ax.scatter(xi, guides, color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
                   s=50, zorder=3, label=model, edgecolors="k", linewidths=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_TITLE.get(r, r) for r in COMMON_REWARDS], rotation=15, fontsize=8)
    ax.set_title("B. Against GEOM-Drugs", fontsize=9)

    handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS[m], color="w", markerfacecolor=MODEL_COLOURS[m],
                          markeredgecolor="k", markersize=8, label=m) for m in models]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.35, label="null / prior reference"))
    handles.append(plt.Line2D([0], [0], marker=MODEL_MARKERS["Quetzal"], color="w",
                              markerfacecolor="none", markeredgecolor=MODEL_COLOURS["Quetzal"],
                              markeredgewidth=1.3, markersize=8,
                              label="Quetzal, independent reference (panel A only)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp06_nn_similarity.pdf")


if __name__ == "__main__":
    main()
