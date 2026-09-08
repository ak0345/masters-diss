"""make_fig15_chemspace.py -- Figure 15 analogue: nearest-neighbour distance
and 2D embedding of guided samples relative to the frozen prior and
GEOM-Drugs, mirroring ../../figures/make_fig15_chemspace.py's own panels A-C
(panel D, bond-perception artifacts, is specific to Quetzal's 3D-to-SMILES
conversion step and has no equivalent here -- MolGPT emits SMILES directly).

  A  nearest-neighbour similarity to the PRIOR (max Tanimoto over ECFP4/Morgan,
     radius 2, 2048 bits), as an ECDF per reward's guided pool, with an
     independent second prior draw ("null") as the reference curve. A guided
     curve sitting to the RIGHT of the null means guided samples cluster
     closer to chemistry the prior already reaches than two independent
     unguided draws sit to each other -- i.e. the guide concentrates mass
     inside the prior's own reachable set rather than moving away from it.
  B  nearest-neighbour similarity to GEOM-Drugs (the shared training corpus),
     same metric, with the prior's own distance to GEOM as the reference
     curve every guided curve is read against.
  C  2D embedding (UMAP with a PCA-by-SVD fallback -- see `embed()`), present
     for orientation only, not as a quantitative claim (see Quetzal's own
     fig15 docstring for why: distances in a neighbour embedding aren't
     quantities the way an ECDF's x-axis is).

"prior" = base_smiles.txt pooled from every sweep-*-s0 config (seed-0 base
draws); "null" = the same pool from every sweep-*-s42 config -- an
independent second unguided draw of the identical frozen prior, giving the
prior-vs-prior baseline panel A's guided curves are read against, matching
Quetzal's own seed0-vs-seed42 base-draw convention (its `_base/<family>/
seed{0,42}` split serves the same role).

Reads results/molgpt/dumps/sweep-*/{base,guided}_smiles.txt (07_final_dump.sh's
output) and ../../shared_data/geom_drugs_filtered.smi.
"""
import glob
import os
import random
import sys

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import rel, REWARD_COLOURS, savefig, require, plt

DUMPS_DIR = rel("results", "molgpt", "dumps")
GEOM_SMI = rel("..", "shared_data", "geom_drugs_filtered.smi")
N_GEOM = 1000
N_PER_CATEGORY = 400
SEED = 0


def fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_fps(smis, rng, n):
    """(fingerprints, bit-matrix) for up to n valid molecules from smis."""
    if len(smis) > n:
        smis = rng.sample(smis, n)
    fps, bits = [], []
    for s in smis:
        fp = fingerprint(s)
        if fp is None:
            continue
        fps.append(fp)
        v = np.zeros(2048, dtype=np.float32)
        v[list(fp.GetOnBits())] = 1.0
        bits.append(v)
    return fps, (np.array(bits, dtype=np.float32) if bits else np.zeros((0, 2048), dtype=np.float32))


def nn_to(query, ref):
    """Copied from ../../figures/make_fig15_chemspace.py's `nn_to()` -- max
    Tanimoto similarity of each query fingerprint to any fingerprint in ref."""
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref))
                     for f in query], dtype=float)


def embed(X, n_neighbors=25, min_dist=0.15, seed=0):
    """2D embedding: UMAP, falling back to PCA. Returns (Z, xlabel, ylabel).
    Copied from ../../figures/make_fig15_chemspace.py's own `embed()`."""
    n_neighbors = min(n_neighbors, max(len(X) - 1, 1))
    try:
        import umap
        red = umap.UMAP(n_components=2, metric="jaccard",
                        n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed)
        return red.fit_transform(X), "UMAP-1", "UMAP-2"
    except ImportError:
        print("[fig15] umap-learn not installed; falling back to PCA.")
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


def main():
    require(DUMPS_DIR, "bash scripts_molgpt/07_final_dump.sh")
    rng = random.Random(SEED)

    geom_smiles = sample_lines(GEOM_SMI, N_GEOM, rng)
    prior_smiles = pooled(f"{DUMPS_DIR}/sweep-*-s0", "base_smiles.txt")
    null_smiles = pooled(f"{DUMPS_DIR}/sweep-*-s42", "base_smiles.txt")

    rewards = sorted({os.path.basename(d).split("-")[1] for d in glob.glob(f"{DUMPS_DIR}/sweep-*")})
    guided_pool = {r: [] for r in rewards}
    for d in sorted(glob.glob(f"{DUMPS_DIR}/sweep-*")):
        reward = os.path.basename(d).split("-")[1]
        f = os.path.join(d, "guided_smiles.txt")
        if os.path.exists(f):
            guided_pool[reward] += [l.strip() for l in open(f) if l.strip()]

    geom_fps, geom_bits = load_fps(geom_smiles, rng, N_GEOM)
    prior_fps, prior_bits = load_fps(prior_smiles, rng, N_PER_CATEGORY)
    null_fps, null_bits = load_fps(null_smiles, rng, N_PER_CATEGORY)
    guided = {r: load_fps(guided_pool[r], rng, N_PER_CATEGORY) for r in rewards}

    if len(prior_fps) < 10 or any(len(guided[r][0]) < 10 for r in rewards):
        raise SystemExit("[FATAL] too few valid molecules for a chemspace comparison")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---------------- A: nearest neighbour to the prior --------------------
    ax = axes[0]
    d = np.sort(nn_to(null_fps, prior_fps))
    ax.plot(d, np.linspace(0, 1, len(d)), color="k", ls="--", lw=2.0, zorder=5,
            label=f"prior (2nd draw) = null ({d.mean():.3f})")
    for r in rewards:
        f_, _ = guided[r]
        d = np.sort(nn_to(f_, prior_fps))
        ax.plot(d, np.linspace(0, 1, len(d)), color=REWARD_COLOURS.get(r, "#333333"),
                lw=1.5, zorder=2, label=f"guided: {r} ({d.mean():.3f})")
    ax.set_xlabel("max Tanimoto to any prior molecule")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("A. Distance to the prior sample", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    # ---------------- B: nearest neighbour to GEOM --------------------------
    ax = axes[1]
    d = np.sort(nn_to(prior_fps, geom_fps))
    ax.plot(d, np.linspace(0, 1, len(d)), color="0.4", lw=2.0, zorder=5,
            label=f"prior ({d.mean():.3f})")
    for r in rewards:
        f_, _ = guided[r]
        d = np.sort(nn_to(f_, geom_fps))
        ax.plot(d, np.linspace(0, 1, len(d)), color=REWARD_COLOURS.get(r, "#333333"),
                lw=1.5, label=f"guided: {r} ({d.mean():.3f})")
    ax.set_xlabel("max Tanimoto to any GEOM molecule")
    ax.set_title("B. Distance to GEOM-Drugs", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    # ---------------- C: 2D embedding, qualitative only ---------------------
    ax = axes[2]
    mats = [geom_bits, prior_bits] + [guided[r][1] for r in rewards]
    names = ["GEOM-Drugs", "frozen prior (base)"] + [f"guided: {r}" for r in rewards]
    Z, xl, yl = embed(np.vstack(mats), seed=SEED)
    off = 0
    for name, m in zip(names, mats):
        z = Z[off:off + len(m)]; off += len(m)
        if name == "GEOM-Drugs":
            ax.scatter(z[:, 0], z[:, 1], s=8, c="0.7", lw=0, alpha=0.35, zorder=1, label=name)
        elif name.startswith("frozen"):
            ax.scatter(z[:, 0], z[:, 1], s=10, marker="x", c="0.15", alpha=0.7, zorder=2, label=name)
        else:
            r = name.split(": ")[1]
            ax.scatter(z[:, 0], z[:, 1], s=10, lw=0, alpha=0.6, zorder=3,
                       color=REWARD_COLOURS.get(r, "#333333"), label=name)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title("C. Embedding (orientation only)", fontsize=9)
    ax.legend(fontsize=6, markerscale=1.5, loc="best")

    # No suptitle: the caption carries the description.
    savefig(fig, "fig15_chemspace.png")


if __name__ == "__main__":
    main()
