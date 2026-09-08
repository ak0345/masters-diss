#!/usr/bin/env python3
"""Chemistry-space analysis: scaffolds, properties, and nearest-neighbour similarity.

Three statistics, each computed against an explicit null, for every architecture and reward.

WHY THE NULLS MATTER, AND THE SEED TRAP
---------------------------------------
A guided dump and its matched base dump are generated under the *same* sampling seed, so their
random streams are coupled. Comparing them directly makes the guide look far closer to the
prior than it is: on one pilot configuration the scaffold Jaccard was 0.293 against a
prior-vs-prior null of 0.042, purely because 64% of the molecules were literally identical.

Every distributional comparison here therefore uses an INDEPENDENT prior draw as the
reference:

    reference   prior, seed 42
    null        prior seed 0   vs reference
    guided      guided seed 0  vs reference

The same-seed coupling is not discarded. It is reported separately as the identical-molecule
fraction, which is a direct measure of how often the guide changes nothing at all.

OUTPUT
    summary.json beside this file.
"""
import argparse
import glob
import json
import os
import random
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

REWARDS = ["osim", "peri", "fexo", "nitrogen"]
PROPS = ["logP", "TPSA", "MW", "HBA", "HBD"]


def read(path, n, seed=0):
    smis = [l.split()[0] for l in open(path) if l.strip()]
    rng = random.Random(seed)
    return rng.sample(smis, min(n, len(smis)))


def mols(smis):
    out = [Chem.MolFromSmiles(s) for s in smis]
    return [m for m in out if m is not None]


def scaffolds(ms):
    out = []
    for m in ms:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    return out


def bits(ms):
    out = np.zeros((len(ms), 2048), dtype=np.float32)
    for i, m in enumerate(ms):
        for b in GEN.GetFingerprint(m).GetOnBits():
            out[i, b] = 1.0
    return out


def descriptors(ms):
    f = (Descriptors.MolLogP, Descriptors.TPSA, Descriptors.MolWt,
         Descriptors.NumHAcceptors, Descriptors.NumHDonors)
    return np.array([[fn(m) for fn in f] for m in ms], dtype=float)


def max_tanimoto(A, B, block=256):
    """For each row of A, its maximum Tanimoto similarity to any row of B."""
    nb = B.sum(1)
    best = np.empty(len(A))
    for i in range(0, len(A), block):
        a = A[i:i + block]
        inter = a @ B.T
        union = a.sum(1)[:, None] + nb[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            best[i:i + block] = np.where(union > 0, inter / union, 0.0).max(1)
    return best


def scaffold_overlap(ref_scaf, query_scaf):
    R, Q = set(ref_scaf), set(query_scaf)
    return {
        "jaccard": len(R & Q) / max(len(R | Q), 1),
        "reuse_frac": sum(1 for x in query_scaf if x in R) / max(len(query_scaf), 1),
        "n_distinct": len(Q),
    }


def boot_ci(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perm_test(a, b, reps=2000, seed=0):
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs:
            hits += 1
    return float((hits + 1) / (reps + 1))


def analyse(guided_path, prior_same_path, prior_ref_path, n, tag):
    g_s = read(guided_path, n); p_same_s = read(prior_same_path, n); p_ref_s = read(prior_ref_path, n)
    g, p_same, p_ref = mols(g_s), mols(p_same_s), mols(p_ref_s)
    if min(len(g), len(p_same), len(p_ref)) < 100:
        return None

    # 1. identical molecules under the shared sampling seed
    ident = len(set(g_s) & set(p_same_s)) / max(len(set(g_s)), 1)

    # 2. scaffolds, against an independent prior draw
    sg, sp_same, sp_ref = scaffolds(g), scaffolds(p_same), scaffolds(p_ref)
    scaf = {"guided": scaffold_overlap(sp_ref, sg),
            "null": scaffold_overlap(sp_ref, sp_same)}

    # 3. nearest-neighbour similarity, with bootstrap CI and a permutation test
    bg, bp_same, bp_ref = bits(g), bits(p_same), bits(p_ref)
    nn_g = max_tanimoto(bg, bp_ref)
    nn_n = max_tanimoto(bp_same, bp_ref)
    # The same statistic measured the WRONG way, against the guided pool's own seed-matched
    # prior. It is reported so that the size of the seed-coupling bias can be quoted on the
    # same pools and the same scale as the corrected number, rather than from a second
    # pipeline whose sampling differs. It is a measurement of the artefact, not a result.
    nn_coupled = max_tanimoto(bg, bp_same)
    nn = {
        "guided_mean": float(nn_g.mean()), "guided_ci": boot_ci(nn_g),
        "null_mean": float(nn_n.mean()), "null_ci": boot_ci(nn_n),
        "gap": float(nn_g.mean() - nn_n.mean()),
        "p_perm": perm_test(nn_g, nn_n),
        "guided_vs_seed_matched_mean": float(nn_coupled.mean()),
    }

    # 4. property marginals
    dg, dp = descriptors(g), descriptors(p_ref)
    prop = {PROPS[i]: {"guided": [float(dg[:, i].mean()), float(dg[:, i].std())],
                       "prior": [float(dp[:, i].mean()), float(dp[:, i].std())]}
            for i in range(len(PROPS))}

    print("  %-22s ident %.1f%% | scaf J %.3f (null %.3f) | NN %.4f vs %.4f p=%.4f"
          % (tag, 100 * ident, scaf["guided"]["jaccard"], scaf["null"]["jaccard"],
             nn["guided_mean"], nn["null_mean"], nn["p_perm"]))
    return {"identical_frac": ident, "scaffold": scaf, "nn": nn, "properties": prop,
            "n_guided": len(g), "n_prior": len(p_ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "summary.json"))
    args = ap.parse_args()

    res = {}

    print("Quetzal:")
    res["quetzal"] = {}
    for r in REWARDS:
        g = os.path.join(ROOT, "results/dumps",
                         "sweep-%s-hidden-db-replay_off-b10-s0" % r, "seed0", "guided_smiles.txt")
        ps = os.path.join(ROOT, "results/dumps/_base", r, "seed0", "base_smiles.txt")
        pr = os.path.join(ROOT, "results/dumps/_base", r, "seed42", "base_smiles.txt")
        if all(os.path.exists(p) for p in (g, ps, pr)):
            out = analyse(g, ps, pr, args.n, r)
            if out: res["quetzal"][r] = out

    for leg in ("molgpt", "g2pt"):
        print("%s:" % leg)
        res[leg] = {}
        for r in REWARDS:
            base = os.path.join(ROOT, leg, "results", leg, "dumps")
            g = os.path.join(base, "sweep-%s-hidden-db-replay_off-b10-s0" % r, "guided_smiles.txt")
            ps = os.path.join(base, "sweep-%s-hidden-db-replay_off-b10-s0" % r, "base_smiles.txt")
            pr = os.path.join(base, "sweep-%s-hidden-db-replay_off-b10-s42" % r, "base_smiles.txt")
            if all(os.path.exists(p) for p in (g, ps, pr)):
                out = analyse(g, ps, pr, args.n, r)
                if out: res[leg][r] = out

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
