"""compute_geom_baseline.py -- best-of-10,000 reference score, per reward,
on a random sample of the shared filtered GEOM-Drugs corpus
(shared_data/geom_drugs_filtered.smi).

Both MolGPT and G2PT now train on this exact corpus, so it is a fair,
model-independent reference line for the three-way comparison figures --
unlike Quetzal's own GEOM baseline (computed from its own 3D-native GEOM
subset before either pilot existed), this one reference applies unchanged
to all three legs. Run once; output is cached to
`geom_baseline_best_of_10k.json` and read from there by every comparison
figure, not recomputed per figure.
"""
import json
import random

import numpy as np
import scipy
if not hasattr(scipy, "histogram"):
    scipy.histogram = np.histogram
from rdkit import Chem
from guacamol import standard_benchmarks as sb

GEOM_SMI = "/workspace/quetzal/quetzal_gfn/shared_data/geom_drugs_filtered.smi"
OUT_PATH = "/workspace/quetzal/quetzal_gfn/comparison_figures/geom_baseline_best_of_10k.json"
BENCH_FN = {"osim": "hard_osimertinib", "peri": "perindopril_rings", "fexo": "hard_fexofenadine"}
N_SAMPLE = 10000


def nitrogen_frac(mol):
    """Same formula as reward_adapter.py's _score_nitrogen_fraction."""
    heavy = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return 0.0
    n_count = sum(1 for z in heavy if z == 7)
    eps = 0.05
    return (n_count + eps) / (len(heavy) + eps)


def top_k_stats(vals):
    vals = sorted(vals, reverse=True)
    return {"top1": vals[0], "top10": sum(vals[:10]) / 10, "top100": sum(vals[:100]) / 100}


def main():
    random.seed(0)
    with open(GEOM_SMI) as f:
        smiles = [l.strip() for l in f if l.strip()]
    random.shuffle(smiles)
    smiles = smiles[:N_SAMPLE]
    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles]
    mols = [(s, m) for s, m in mols if m is not None]
    print(f"usable: {len(mols)} / {N_SAMPLE}")

    out = {"nitrogen": top_k_stats(nitrogen_frac(m) for _, m in mols)}
    for key, fn_name in BENCH_FN.items():
        obj = getattr(sb, fn_name)().objective
        vals = []
        for s, _m in mols:
            try:
                vals.append(float(obj.score(s)))
            except Exception:
                vals.append(0.0)
        out[key] = top_k_stats(vals)
        print(key, out[key])

    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
