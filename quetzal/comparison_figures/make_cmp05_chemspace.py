"""make_cmp05_chemspace.py -- three-way comparison analogue of each leg's own
fig15: where Quetzal's, MolGPT's, and G2PT's guided samples sit relative to
GEOM-Drugs and each other, in chemical space, one panel per reward common to
all three legs' pilot grids (nitrogen, osim, peri, fexo).

Single UMAP embedding (metric="jaccard" on ECFP4/Morgan, radius 2, 2048 bits
-- see g2pt/figures/make_fig15_chemspace.py's `embed()` for the fallback and
full rationale) fit once on the union of every point across every reward and
model, so a panel-to-panel comparison reads off a shared coordinate system.
Only "guided" samples are shown per model/reward (not "base") -- each leg's
own per-leg fig15 already shows guided vs. base vs. GEOM-Drugs for that one
model; this figure's job is comparing the three models' guided distributions
against each other, so adding each model's base pool on top would triple the
category count per panel for a comparison this figure isn't making.

Quetzal's rows are restricted to `sweep-*`, guide in {base,hidden}, objective
in {db,rtb}, replay=off, beta=10 -- the exact grid the pilots share with it
(same filter as make_cmp01_landscape.py's `quetzal_rows()`). Reads guided
SMILES directly from each matching config's dump directory rather than from
master_table.csv, since chemspace needs the raw molecules, not the
aggregated scores.
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
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def to_array(fps):
    arr = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        arr[i, list(fp.GetOnBits())] = 1
    return arr


def sample_lines(path, n, rng):
    if not os.path.exists(path):
        return []
    lines = [l.strip() for l in open(path) if l.strip()]
    return rng.sample(lines, n) if len(lines) > n else lines


def embed(X, n_neighbors=25, min_dist=0.15, seed=0):
    """2D embedding: UMAP, falling back to PCA. Copied from
    g2pt/figures/make_fig15_chemspace.py's `embed()` -- see that file for
    the full rationale (Jaccard matches Tanimoto on binary bits)."""
    n_neighbors = min(n_neighbors, max(len(X) - 1, 1))
    try:
        import umap
        red = umap.UMAP(n_components=2, metric="jaccard",
                        n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed)
        return red.fit_transform(X), "UMAP-1", "UMAP-2"
    except ImportError:
        print("[cmp05] umap-learn not installed; falling back to PCA.")
        Xc = X.astype(np.float64)
        Xc = Xc - Xc.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / max((S ** 2).sum(), 1e-12)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Z = Xc @ Vt[:2].T
        if not np.all(np.isfinite(Z)):
            raise RuntimeError("PCA produced non-finite coordinates")
        return (Z, f"PC1 ({100*var[0]:.1f}% var)",
                f"PC2 ({100*var[1]:.1f}% var)")


def quetzal_guided_pool(rng):
    """{reward: [smiles]}, pooled across every sweep dir matching the shared
    grid (guide in {base,hidden}, objective in {db,rtb}, replay=off,
    beta=10), reading each config's own seed0/guided_smiles.txt directly."""
    pool = {r: [] for r in COMMON_REWARDS}
    for d in sorted(glob.glob(f"{DUMPS_DIRS['Quetzal']}/sweep-*")):
        m = QUETZAL_NAME_RE.match(os.path.basename(d))
        if not m or m.group("replay") != "off" or m.group("beta") != "10":
            continue
        reward = m.group("reward")
        if reward not in COMMON_REWARDS:
            continue
        f = os.path.join(d, "seed0", "guided_smiles.txt")
        if os.path.exists(f):
            pool[reward] += [l.strip() for l in open(f) if l.strip()]
    return {r: (rng.sample(v, min(N_PER_CATEGORY, len(v))) if v else [])
            for r, v in pool.items()}


def pilot_guided_pool(model, rng):
    """Same shape, for MolGPT/G2PT: their whole sweep grid is already this
    comparable grid (no beta sweep, no extra reward), so no filter needed
    beyond reading guided_smiles.txt from every sweep-* dump directory."""
    pool = {r: [] for r in COMMON_REWARDS}
    for d in sorted(glob.glob(f"{DUMPS_DIRS[model]}/sweep-*")):
        reward = os.path.basename(d).split("-")[1]
        if reward not in COMMON_REWARDS:
            continue
        f = os.path.join(d, "guided_smiles.txt")
        if os.path.exists(f):
            pool[reward] += [l.strip() for l in open(f) if l.strip()]
    return {r: (rng.sample(v, min(N_PER_CATEGORY, len(v))) if v else [])
            for r, v in pool.items()}


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(DUMPS_DIRS[model], f"produce {model}'s dumps first")

    rng = random.Random(SEED)
    geom_smiles = sample_lines(GEOM_SMI, N_GEOM, rng)
    guided = {
        "Quetzal": quetzal_guided_pool(rng),
        "MolGPT": pilot_guided_pool("MolGPT", rng),
        "G2PT": pilot_guided_pool("G2PT", rng),
    }
    for model, pool in guided.items():
        print(f"[cmp05] {model} pooled guided n = " +
              ", ".join(f"{r}={len(v)}" for r, v in pool.items()))

    # categories: (label, smiles list, colour, marker)
    categories = [("GEOM-Drugs", geom_smiles, "0.6", ".")]
    for model in ("Quetzal", "MolGPT", "G2PT"):
        for reward in COMMON_REWARDS:
            categories.append((f"{model}:{reward}", guided[model][reward],
                               MODEL_COLOURS[model], MODEL_MARKERS[model]))

    all_fps, all_labels = [], []
    for label, smis, _, _ in categories:
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
                 for label, *_ in categories}

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, reward in zip(axes.flat, COMMON_REWARDS):
        geom_idx = label_idx["GEOM-Drugs"]
        ax.scatter(coords[geom_idx, 0], coords[geom_idx, 1], s=10, alpha=0.25,
                   color="0.6", marker=".", label=f"GEOM-Drugs (n={len(geom_idx)})",
                   zorder=1)
        for model in ("Quetzal", "MolGPT", "G2PT"):
            idx = label_idx[f"{model}:{reward}"]
            if not idx:
                continue
            ax.scatter(coords[idx, 0], coords[idx, 1], s=20, alpha=0.7,
                       color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model],
                       label=f"{model} (n={len(idx)})", zorder=3, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(REWARD_TITLE.get(reward, reward))
        ax.legend(fontsize=7, loc="best", markerscale=1.5)

    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp05_chemspace.png")


if __name__ == "__main__":
    main()
