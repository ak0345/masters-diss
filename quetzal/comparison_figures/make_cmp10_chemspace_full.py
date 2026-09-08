"""make_cmp10_chemspace_full.py -- the fullest chemspace comparison: for
each of the three models, GEOM-Drugs (the shared training corpus) vs. the
frozen prior's own unconditional samples vs. DB/RTB-guided samples vs.
FUDGE-guided samples, one panel per common reward, in a single joint UMAP
embedding.

Extends make_cmp05_chemspace.py (DB/RTB guided only, no prior/FUDGE) with
the two categories added since: the frozen prior itself (so "does guidance
move samples away from the prior's own unconditional cloud" is answerable
directly, not just "do the two guided clouds differ from each other"), and
FUDGE (the second, differently-trained guide baseline -- see decisions.md's
"FUDGE baseline" entries). Colour = model (as in every other comparison
figure here); marker SHAPE = category (prior / DB-RTB-guided / FUDGE-guided)
-- a different visual channel from make_cmp05's model-only marker
convention, since this figure has a second axis (which guide, if any) to
show that cmp05 didn't need.

Base/prior pools are reward-agnostic for MolGPT/G2PT (unconditional
generation doesn't depend on which reward a config targets) but
reward-specific for Quetzal (`_base/<reward>/seed0`) -- same convention as
every other script here that pools base samples. FUDGE pools are a single
run per reward per model (no sweep to pool across, unlike DB/RTB's 12
matching configs per reward).
"""
import glob
import os
import random
import re
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, savefig, require, plt,
                               MODEL_COLOURS, COMMON_REWARDS,
                               REWARD_TITLE, DUMPS_DIRS)
from make_cmp05_chemspace import (fingerprint, to_array, sample_lines, embed,
                                   quetzal_guided_pool, pilot_guided_pool,
                                   QUETZAL_NAME_RE, GEOM_SMI, N_GEOM, N_PER_CATEGORY, SEED)

FUDGE_DUMPS_DIRS = {
    "Quetzal": rel("results", "fudge_dumps"),
    "MolGPT": rel("molgpt", "results", "molgpt", "fudge_dumps"),
    "G2PT": rel("g2pt", "results", "g2pt", "fudge_dumps"),
}
TYPE_MARKERS = {"prior": "x", "DB/RTB": "o", "FUDGE": "^"}
TYPE_ALPHA = {"prior": 0.45, "DB/RTB": 0.7, "FUDGE": 0.7}


def quetzal_prior_pool(rng):
    pool = {}
    for r in COMMON_REWARDS:
        f = rel("results", "dumps", "_base", r, "seed0", "base_smiles.txt")
        lines = sample_lines(f, N_PER_CATEGORY, rng)
        pool[r] = lines
    return pool


def pilot_prior_pool(model, rng):
    base_pool = []
    for d in sorted(glob.glob(f"{DUMPS_DIRS[model]}/sweep-*")):
        f = os.path.join(d, "base_smiles.txt")
        if os.path.exists(f):
            base_pool += [l.strip() for l in open(f) if l.strip()]
    sampled = rng.sample(base_pool, min(N_PER_CATEGORY, len(base_pool))) if base_pool else []
    return {r: sampled for r in COMMON_REWARDS}


def fudge_guided_pool(model, rng):
    pool = {}
    for r in COMMON_REWARDS:
        f = os.path.join(FUDGE_DUMPS_DIRS[model], r, "guided_smiles.txt")
        pool[r] = sample_lines(f, N_PER_CATEGORY, rng)
    return pool


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(DUMPS_DIRS[model], f"produce {model}'s DB/RTB dumps first")
        require(FUDGE_DUMPS_DIRS[model], f"produce {model}'s FUDGE dumps first")

    rng = random.Random(SEED)
    geom_smiles = sample_lines(GEOM_SMI, N_GEOM, rng)

    prior = {"Quetzal": quetzal_prior_pool(rng),
             "MolGPT": pilot_prior_pool("MolGPT", rng),
             "G2PT": pilot_prior_pool("G2PT", rng)}
    dbrtb = {"Quetzal": quetzal_guided_pool(rng),
              "MolGPT": pilot_guided_pool("MolGPT", rng),
              "G2PT": pilot_guided_pool("G2PT", rng)}
    fudge = {m: fudge_guided_pool(m, rng) for m in ("Quetzal", "MolGPT", "G2PT")}

    for model in ("Quetzal", "MolGPT", "G2PT"):
        print(f"[cmp10] {model} prior n=" + ",".join(f"{r}={len(prior[model][r])}" for r in COMMON_REWARDS))
        print(f"[cmp10] {model} DB/RTB n=" + ",".join(f"{r}={len(dbrtb[model][r])}" for r in COMMON_REWARDS))
        print(f"[cmp10] {model} FUDGE  n=" + ",".join(f"{r}={len(fudge[model][r])}" for r in COMMON_REWARDS))

    categories = [("GEOM-Drugs", geom_smiles)]
    pools = {"prior": prior, "DB/RTB": dbrtb, "FUDGE": fudge}
    for model in ("Quetzal", "MolGPT", "G2PT"):
        for kind in ("prior", "DB/RTB", "FUDGE"):
            for reward in COMMON_REWARDS:
                categories.append((f"{model}:{kind}:{reward}", pools[kind][model][reward]))

    all_fps, all_labels = [], []
    for label, smis in categories:
        for s in smis:
            fp = fingerprint(s)
            if fp is not None:
                all_fps.append(fp)
                all_labels.append(label)

    if len(all_fps) < 10:
        raise SystemExit("[FATAL] too few valid molecules to fit a joint chemspace embedding")

    X = to_array(all_fps)
    coords, xlabel, ylabel = embed(X, seed=SEED)
    label_idx = {label: [i for i, l in enumerate(all_labels) if l == label]
                 for label, _ in categories}

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, reward in zip(axes.flat, COMMON_REWARDS):
        geom_idx = label_idx["GEOM-Drugs"]
        ax.scatter(coords[geom_idx, 0], coords[geom_idx, 1], s=8, alpha=0.2,
                   color="0.6", marker=".", label=f"GEOM-Drugs (n={len(geom_idx)})", zorder=1)
        for model in ("Quetzal", "MolGPT", "G2PT"):
            for kind in ("prior", "DB/RTB", "FUDGE"):
                idx = label_idx[f"{model}:{kind}:{reward}"]
                if not idx:
                    continue
                ax.scatter(coords[idx, 0], coords[idx, 1], s=22, alpha=TYPE_ALPHA[kind],
                           color=MODEL_COLOURS[model], marker=TYPE_MARKERS[kind],
                           label=f"{model} {kind} (n={len(idx)})",
                           edgecolors="none" if kind != "prior" else None, zorder=2 if kind == "prior" else 3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(REWARD_TITLE.get(reward, reward))
        ax.legend(fontsize=6, loc="best", markerscale=1.3, ncol=2)

    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp10_chemspace_full.png")


if __name__ == "__main__":
    main()
