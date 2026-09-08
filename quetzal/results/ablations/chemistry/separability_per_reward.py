#!/usr/bin/env python3
"""Per-reward k-NN separability, one-vs-rest.

Table 3.4 asks one pooled question: can a classifier name which reward family a molecule's
guide was trained against? It answers 0.203 against a 0.200 majority baseline, i.e. no.

A pooled number can hide a single separable family among three inseparable ones, so this
repeats the test one-vs-rest: for each reward, can a classifier tell that family apart from
the other three combined? The baseline is then the majority rate for that split, and a
shuffled-label null is computed alongside it.
"""
import json, os, random
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REWARDS = ["osim", "peri", "fexo", "nitrogen"]


def bits(path, n, seed=0):
    smis = [l.split()[0] for l in open(path) if l.strip()]
    smis = random.Random(seed).sample(smis, min(n, len(smis)))
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v = np.zeros(2048, dtype=np.float32)
        for b in GEN.GetFingerprint(m).GetOnBits():
            v[b] = 1.0
        out.append(v)
    return np.array(out, dtype=np.float32)


def main(n=1200, k=15, folds=5):
    X, y = [], []
    for i, r in enumerate(REWARDS):
        p = os.path.join(ROOT, "results/dumps",
                         "sweep-%s-hidden-db-replay_off-b10-s0" % r, "seed0", "guided_smiles.txt")
        if not os.path.exists(p):
            continue
        b = bits(p, n)
        X.append(b); y += [i] * len(b)
    X = np.vstack(X); y = np.array(y)
    rng = np.random.default_rng(0)
    res = {}
    for i, r in enumerate(REWARDS):
        yb = (y == i).astype(int)
        clf = KNeighborsClassifier(n_neighbors=k, metric="jaccard")
        acc = cross_val_score(clf, X, yb, cv=folds).mean()
        maj = max(yb.mean(), 1 - yb.mean())
        null = cross_val_score(clf, X, rng.permutation(yb), cv=folds).mean()
        res[r] = {"accuracy": float(acc), "majority": float(maj), "shuffled_null": float(null)}
        print("  %-9s acc %.3f | majority %.3f | shuffled null %.3f%s"
              % (r, acc, maj, null, "   <- above both" if acc > max(maj, null) + 0.005 else ""))
    out = os.path.join(HERE, "separability.json")
    json.dump(res, open(out, "w"), indent=2, sort_keys=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
