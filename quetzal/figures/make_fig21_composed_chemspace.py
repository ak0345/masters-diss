#!/usr/bin/env python3
"""
make_fig21_composed_chemspace.py -- Figure 21: where the composed sampler's
molecules actually land.

The composition track mixes independently trained per-component guides at
sampling time. Its terminal scores sit close to the frozen prior's, but that
alone cannot say whether the composed sampler reaches the same chemistry as a
single-objective guide or somewhere else.

  A, B  a joint 2D embedding (UMAP on ECFP4 with the Jaccard metric, which is
        Tanimoto on binary vectors) per benchmark. GEOM-Drugs and the frozen
        prior are a grey backdrop; the single-objective guides and the three
        composed operators are drawn over them. Orientation only: a neighbour
        embedding's distances are not quantities.

  C     the quantitative version. For each sample, the maximum Tanimoto
        similarity to a held-out draw from the frozen prior, averaged over
        molecules. The unguided-vs-unguided null is the same statistic with a
        fresh unguided sample in place of the guided one, and it is what says
        whether any separation is real.

INPUTS
  reference/geom_drugs_smiles.txt
  results/dumps/sweep-<fam>-*/seed0/guided_smiles.txt
  results/dumps_composed/<reward>/<arch>-<obj>-s<seed>/<op>/seed0/composed_smiles.txt
                                                              .../base_smiles.txt

USAGE
  python figures/make_fig21_composed_chemspace.py --out out/fig21.pdf
"""
import argparse
import glob
import os
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

REWARDS = [("osimertinib", "osim", "Osimertinib MPO"),
           ("perindopril", "peri", "Perindopril MPO")]
OPERATORS = ["linear", "product", "harmonic"]
OP_COLOUR = {"linear": "#4C72B0", "product": "#DD8452", "harmonic": "#55A868"}
GUIDE_COLOUR = {"base": "#8C8C8C", "tempgain": "#C44E52", "hidden": "#8172B3"}
GUIDE_LABEL = {"base": "residual", "tempgain": "temp-gain", "hidden": "hidden"}


def bits(path, n, rng):
    """Fingerprint bit matrix from a SMILES file, subsampled to n."""
    smis = [l.strip().split()[0] for l in open(path) if l.strip()]
    if len(smis) > n:
        smis = rng.sample(smis, n)
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = np.zeros(2048, dtype=np.float32)
        for b in _GEN.GetFingerprint(m).GetOnBits():
            v[b] = 1.0
        out.append(v)
    return np.array(out, dtype=np.float32)


def mean_max_tanimoto(A, B, block=256):
    """Mean over rows of A of the max Tanimoto similarity to any row of B."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    nb = B.sum(1)
    best = np.empty(len(A), dtype=np.float64)
    for i in range(0, len(A), block):
        a = A[i:i + block]
        inter = a @ B.T
        union = a.sum(1)[:, None] + nb[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(union > 0, inter / union, 0.0)
        best[i:i + block] = t.max(1)
    return float(best.mean())


def composed_paths(root, reward, op):
    return sorted(glob.glob(os.path.join(
        root, reward, "*", op, "seed*", "composed_smiles.txt")))


def embed(X, seed=0):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import umap
            red = umap.UMAP(n_components=2, metric="jaccard", n_neighbors=25,
                            min_dist=0.15, random_state=seed)
            return red.fit_transform(X)
    except Exception as exc:                                # pragma: no cover
        print("[fig21] UMAP unavailable (%s); falling back to PCA" % exc)
        Xc = X.astype(np.float64)
        Xc = Xc - Xc.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Xc @ Vt[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composed-root", default=fs.rel("results", "dumps_composed"))
    ap.add_argument("--dumps-root", default=fs.rel("results", "dumps"))
    ap.add_argument("--geom", default=fs.rel("reference", "geom_drugs_smiles.txt"))
    ap.add_argument("--n", type=int, default=700, help="molecules per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/fig21.pdf")
    args = ap.parse_args()

    fs.use_paper_style()
    rng = random.Random(args.seed)

    geom = bits(fs.need(args.geom), args.n, rng)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))

    nn_rows = []
    for ax, (long_name, short, title) in zip(axes[:2], REWARDS):
        groups, labels, colours = [], [], []

        # frozen prior: the matched unguided draw recorded beside every composed dump
        prior_paths = sorted(glob.glob(os.path.join(
            args.composed_root, long_name, "*", "*", "seed*", "base_smiles.txt")))
        prior = bits(prior_paths[0], 2 * args.n, rng)
        prior_ref, prior_probe = prior[:len(prior) // 2], prior[len(prior) // 2:]

        # single-objective guides feed panel C only; drawing all six groups over one
        # another turns the embedding into mud.
        for arch in ["base", "tempgain", "hidden"]:
            hits = sorted(glob.glob(os.path.join(
                args.dumps_root, "sweep-%s-%s-db-replay_off-b10-s0" % (short, arch),
                "seed0", "guided_smiles.txt")))
            if not hits:
                continue
            nn_rows.append((title, GUIDE_LABEL[arch],
                            mean_max_tanimoto(bits(hits[0], args.n, rng), prior_ref)))

        # composed samplers, one dump per operator
        for op in OPERATORS:
            paths = composed_paths(args.composed_root, long_name, op)
            if not paths:
                continue
            X = bits(paths[0], args.n, rng)
            groups.append(X); labels.append("composed: " + op); colours.append(OP_COLOUR[op])
            nn_rows.append((title, "composed: " + op, mean_max_tanimoto(X, prior_ref)))

        nn_rows.append((title, "unguided null", mean_max_tanimoto(prior_probe, prior_ref)))

        # joint embedding: backdrop first so the guided groups draw over it
        backdrop = [geom, prior_probe]
        X = np.vstack(backdrop + groups)
        Z = embed(X, seed=args.seed)
        off = 0
        # Backdrop as a density field rather than a point cloud: with several
        # thousand overlapping points, scatter-over-scatter hides whichever group
        # is drawn first and reads as one undifferentiated blob.
        nb = sum(len(a) for a in backdrop)
        ax.hexbin(Z[:nb, 0], Z[:nb, 1], gridsize=34, cmap="Greys",
                  mincnt=1, linewidths=0, zorder=1)
        off = nb
        for arr, colour, label in zip(groups, colours, labels):
            ax.scatter(Z[off:off + len(arr), 0], Z[off:off + len(arr), 1], s=7,
                       c=colour, alpha=0.55, linewidths=0, label=label, zorder=2)
            off += len(arr)
        ax.scatter([], [], s=28, c="#9A9A9A", label="GEOM-Drugs + prior (density)")
        ax.set_title(title)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=7, markerscale=2.2, loc="lower left", ncol=1)

    # ---- panel C: nearest-neighbour similarity to the frozen prior
    ax = axes[2]
    order = ["residual", "temp-gain", "hidden", "composed: linear",
             "composed: product", "composed: harmonic"]
    width = 0.38
    for j, (_, short, title) in enumerate(REWARDS):
        vals = {lab: v for (t, lab, v) in nn_rows if t == title}
        null = vals.get("unguided null", np.nan)
        xs = np.arange(len(order)) + (j - 0.5) * width
        ax.bar(xs, [vals.get(o, np.nan) for o in order], width,
               label=title, alpha=0.85,
               color="#4C72B0" if j == 0 else "#DD8452", zorder=2)
        ax.hlines(null, xs[0] - width, xs[-1] + width,
                  color="#4C72B0" if j == 0 else "#DD8452", ls="--", lw=1.3, zorder=3)
    finite = [v for (_, lab, v) in nn_rows if np.isfinite(v)]
    lo, hi = min(finite), max(finite)
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("mean max Tanimoto to prior")
    ax.set_title("Nearest-neighbour similarity")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    for t, lab, v in nn_rows:
        print("[fig21] %-18s %-20s mean max Tanimoto %.4f" % (t, lab, v))


if __name__ == "__main__":
    main()
