#!/usr/bin/env python3
"""Does the composition weighting choose WHERE the sampler sits, or HOW FAR it can reach?

Reads the sweep `scripts/sweep_peri_weights.sh` produces. No GPU.

Perindopril's objective has exactly two live components, so the weight simplex
(omega_0, omega_1) with omega_1 = 1 - omega_0 is a line, and eleven points cover it
completely.

The reviewer's concern is that uniform weighting might be a poor mix, and that some
interior weighting would reach scores the uniform point cannot. Two different claims hide
in that, and this script separates them:

  TRADE-OFF     moving omega moves the two sub-scores in opposite directions. Expected, and
                not by itself interesting: it says the weights do what weights do.

  EXPANSION     some omega puts molecules somewhere no other omega reaches, so the swept
                family covers more chemistry than any single setting. This is the claim
                that would matter, because the dissertation's conclusion is that guidance
                re-ranks within the frozen prior's reachable set rather than leaving it.

Trade-off is read off the sub-score frontier (c0 against c1). Expansion is measured against
chemistry, not against score, in two ways:

  1. Per point, nearest-neighbour Tanimoto to an INDEPENDENT prior draw, and the fraction of
     Bemis-Murcko scaffolds absent from that draw. The reference is the seed-42 prior dump,
     never the point's own `base_smiles.txt`: a composed dump and the base dump beside it
     share a sampling seed and therefore share molecules outright, which inflates any
     similarity-to-the-other-sample statistic. See the seed-coupling note in the top-level
     README.

  2. Each of those against a MATCHED CONTROL: chunks of the same size drawn from the unguided
     prior, scored against the same reference. A guided point that sits where a prior chunk
     sits has not moved anywhere the prior does not already go.

A third measure, the union of novel scaffolds pooled over the omega line against an
equal-sized prior pool, was tried and is NOT reported. Every omega point is sampled at
seed 0, so adjacent points share around 89% of their molecules outright (see
`adjacent_omega_jaccard` below) and the union across omega is deflated by the shared random
stream rather than by chemistry. It would have understated reach for a reason that has
nothing to do with the weights. The pairwise overlap is reported instead, as the descriptive
fact it is, with the coupling stated rather than read as an effect.

    python scripts/analyse_peri_weights.py
"""
import glob
import json
import os
import random
import sys

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SWEEP = os.path.join(REPO, "results", "ablations", "peri-weights")
# Independent of the sweep's own seed-0 sampling: this is the reference
# `results/ablations/chemistry/analyse_chemistry.py` uses, for the same reason.
REFERENCE = os.path.join(REPO, "results", "dumps", "_base", "peri", "seed42", "base_smiles.txt")
PRIOR_SEED0 = os.path.join(REPO, "results", "dumps", "_base", "peri", "seed0", "base_smiles.txt")
OPERATORS = ("linear", "product", "harmonic")
N_SAMPLE = 300          # molecules per point entering the Tanimoto statistics
N_REF = 1500            # reference molecules
SEED = 0


def read_smiles(path):
    if not os.path.exists(path):
        return []
    return [l.strip() for l in open(path) if l.strip()]


def mols(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(m)
    return out


def fps(ms):
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in ms]


def scaffolds(ms):
    out = set()
    for m in ms:
        try:
            out.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    out.discard("")
    return out


def nn_similarity(query_fps, ref_fps):
    if not query_fps or not ref_fps:
        return float("nan")
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(f, ref_fps))
                          for f in query_fps]))


# ---------------------------------------------------------------- hypervolume
#
# The composed dumps already carry a `hypervolume` field, and it is not the one this
# question needs. That field is computed over three axes, `peri_MPO` together with `c0`
# and `c1`, and `peri_MPO` is a deterministic function of the other two, so the third axis
# is redundant and the volume is Monte-Carlo estimated rather than exact. Worse for a sweep,
# its score floors are refit on each invocation's own pooled rewards, so two omega points
# are normalised differently and their values cannot be compared.
#
# What follows recomputes the front over the two real objectives only, in score space, with
# one reference point fixed for every point in the sweep. That makes the numbers comparable
# across omega, which is the entire purpose.

REF = (0.0, 0.0)      # anti-ideal; GuacaMol component scores live in [0, 1]


def component_scorers(bench_key="perindopril"):
    """The two leaf scorers of the benchmark, on SMILES."""
    from types import SimpleNamespace
    # Modules at the repository root import each other by bare name, so the root has to be
    # on the path before `reward_fn` resolves. See the import note in the top-level README.
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from reward_fn import build_reward_smiles
    out = []
    for idx in (0, 1):
        rc = SimpleNamespace(reward="guacamol_component", reward_benchmark=bench_key,
                             reward_component=idx, invalid_logr=-5.0,
                             force_method="xtb", dataset="geom", reward_beta=1.0)
        out.append(build_reward_smiles(rc))
    return out


def objective_pairs(smis, scorers):
    """Per-molecule (c0, c1) in score space. The rewards are log(score), so exp inverts
    them exactly; an unparseable molecule sits at exp(-5), just above the reference."""
    f0, f1 = scorers
    pts = []
    for s in smis:
        try:
            pts.append((float(np.exp(f0(s))), float(np.exp(f1(s)))))
        except Exception:
            continue
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def pareto_front_2d(objs):
    """Indices of the non-dominated points, maximising both objectives."""
    if len(objs) == 0:
        return objs
    order = np.argsort(-objs[:, 0], kind="stable")
    best_y, keep = -np.inf, []
    for i in order:
        if objs[i, 1] > best_y:
            keep.append(i)
            best_y = objs[i, 1]
    return objs[keep]


def hypervolume_2d(objs, ref=REF):
    """Exact dominated area above `ref`, maximising both objectives."""
    if len(objs) == 0:
        return 0.0, 0
    inside = objs[np.all(objs > np.asarray(ref), axis=1)]
    if len(inside) == 0:
        return 0.0, 0
    front = pareto_front_2d(inside)
    order = np.argsort(-front[:, 0], kind="stable")
    hv, prev_y = 0.0, ref[1]
    for x, y in front[order]:
        if y <= prev_y:
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(hv), int(len(front))


def main():
    rng = random.Random(SEED)

    ref_smis = read_smiles(REFERENCE)
    if not ref_smis:
        sys.exit("missing reference dump: %s" % REFERENCE)
    ref_ms = mols(rng.sample(ref_smis, min(N_REF, len(ref_smis))))
    ref_fps, ref_scaf = fps(ref_ms), scaffolds(ref_ms)

    scorers = component_scorers()
    points, missing = [], []
    for d in sorted(glob.glob(os.path.join(SWEEP, "w*"))):
        if not os.path.isdir(d):
            continue
        w0 = float(os.path.basename(d)[1:])
        for op in OPERATORS:
            summ = os.path.join(d, op, "seed0", "dump_summary.json")
            if not os.path.exists(summ):
                missing.append(os.path.relpath(summ, REPO))
                continue
            s = json.load(open(summ))
            comp, base = s["component_means_composed"], s["component_means_base"]

            smis = read_smiles(os.path.join(d, op, "seed0", "composed_smiles.txt"))
            base_smis = read_smiles(os.path.join(d, op, "seed0", "base_smiles.txt"))
            comp_obj = objective_pairs(smis, scorers)
            base_obj = objective_pairs(base_smis, scorers)
            hv_c, np_c = hypervolume_2d(comp_obj)
            hv_b, np_b = hypervolume_2d(base_obj)

            all_ms = mols(smis)
            # Overlap between omega points is computed on the whole dump; subsampling first
            # would deflate it by the subsample, not by the weights.
            all_canon = [Chem.MolToSmiles(m) for m in all_ms]
            ms = rng.sample(all_ms, N_SAMPLE) if len(all_ms) > N_SAMPLE else all_ms
            scaf = scaffolds(ms)
            novel = scaf - ref_scaf

            points.append({
                "w0": w0, "w1": round(1.0 - w0, 3), "operator": op,
                "c0": comp["c0"], "c1": comp["c1"],
                "c0_base": base["c0"], "c1_base": base["c1"],
                "peri_mpo_mean": comp["peri_MPO"], "peri_mpo_mean_base": base["peri_MPO"],
                "top10": s["guided"]["log_reward_top10"],
                "top10_base": s["base"]["log_reward_top10"],
                "top10_delta": s["top10_delta_guided_minus_base"],
                "parse_rate": s["guided"]["parse_rate"],
                "fcd_vs_base": s.get("fcd", {}).get("guided_vs_base"),
                "n_scored": len(ms),
                "nn_to_independent_prior": nn_similarity(fps(ms), ref_fps),
                "n_scaffolds": len(scaf),
                "novel_scaffold_frac": (len(novel) / len(scaf)) if scaf else float("nan"),
                "hv_composed": hv_c, "hv_base": hv_b,
                "hv_delta": hv_c - hv_b,
                "pareto_count_composed": np_c, "pareto_count_base": np_b,
                "_novel": sorted(novel),
                "_obj": comp_obj,
                "_smis": all_canon,
            })

    if not points:
        sys.exit("no completed sweep points under %s" % SWEEP)

    # ---- how much of the sample actually changes when omega moves ----
    # Descriptive only. Every point shares sampling seed 0, so a high overlap is partly the
    # shared random stream and partly the policies being close. It is reported to make the
    # coupling visible, not read as an effect size.
    overlap = {}
    for op in OPERATORS:
        pts = sorted([p for p in points if p["operator"] == op], key=lambda p: p["w0"])
        js = []
        for a, b in zip(pts, pts[1:]):
            sa, sb = set(a["_smis"]), set(b["_smis"])
            if sa and sb:
                js.append(len(sa & sb) / len(sa | sb))
        if js:
            overlap[op] = {"n_pairs": len(js), "mean_jaccard": float(np.mean(js)),
                           "min_jaccard": float(min(js)), "max_jaccard": float(max(js))}

    # ---- matched control: equal-sized chunks of the unguided prior, same reference ----
    prior_smis = read_smiles(PRIOR_SEED0)
    control = {}
    if prior_smis:
        n_chunks = 8
        pool = rng.sample(prior_smis, min(n_chunks * N_SAMPLE, len(prior_smis)))
        chunks = [pool[i * N_SAMPLE:(i + 1) * N_SAMPLE] for i in range(n_chunks)]
        nn_c, novel_c = [], []
        for c in chunks:
            ms = mols(c)
            if len(ms) < 20:
                continue
            scaf = scaffolds(ms)
            nn_c.append(nn_similarity(fps(ms), ref_fps))
            if scaf:
                novel_c.append(len(scaf - ref_scaf) / len(scaf))
        if nn_c:
            control = {
                "n_chunks": len(nn_c), "chunk_size": N_SAMPLE,
                "nn_to_independent_prior_mean": float(np.mean(nn_c)),
                "nn_to_independent_prior_sd": float(np.std(nn_c, ddof=1)) if len(nn_c) > 1 else 0.0,
                "nn_range": [float(min(nn_c)), float(max(nn_c))],
                "novel_scaffold_frac_mean": float(np.mean(novel_c)),
                "novel_scaffold_frac_range": [float(min(novel_c)), float(max(novel_c))],
            }

    # Does sweeping omega attain a front that no single omega attains? Linear
    # scalarisation can only reach the convex hull of a front, so this is the place that
    # limitation would show. Pooling shares the seed-coupling caveat above, and coupling
    # can only DEFLATE a union, so a union that fails to exceed the best single point is
    # the weaker direction of evidence and is reported as such.
    hv_pooled = {}
    for op in OPERATORS:
        pts = [p for p in points if p["operator"] == op]
        if not pts:
            continue
        allobj = np.vstack([p["_obj"] for p in pts if len(p["_obj"])])
        hv_u, np_u = hypervolume_2d(allobj)
        best = max(p["hv_composed"] for p in pts)
        hv_pooled[op] = {
            "hv_union_over_omega": hv_u, "pareto_count_union": np_u,
            "hv_best_single_omega": best,
            "union_over_best_single": (hv_u / best) if best else float("nan"),
        }

    # NO significance test here, deliberately. The omega points share a sampling seed and
    # most of their molecules, and the hypervolumes collapse to a handful of distinct values,
    # so a sign test over 33 "configurations" would be counting one outcome repeated many
    # times as many independent trials. An earlier version reported 19/33 at p=0.49; that
    # number was pseudo-replication and is withdrawn. It also argued the wrong way round,
    # since failing to reject a null is not evidence for it. What replaces it is a bounded
    # containment statement, which needs no test: how many distinct populations the sweep
    # actually produced, and whether their hypervolumes stay inside the prior's own range.
    hvd = np.array([p["hv_delta"] for p in points])
    distinct = sorted({(round(p["hv_composed"], 6), round(p["hv_base"], 6)) for p in points})
    comp_lo = min(p["hv_composed"] for p in points)
    comp_hi = max(p["hv_composed"] for p in points)
    base_lo = min(p["hv_base"] for p in points)
    base_hi = max(p["hv_base"] for p in points)
    hv_paired = {
        "n_configurations": int(len(hvd)),
        "n_distinct_populations": int(len(distinct)),
        "composed_range_inside_prior_range": bool(comp_lo > base_lo and comp_hi < base_hi),
        "largest_gain": float(hvd.max()), "largest_loss": float(hvd.min()),
        "mean_delta": float(hvd.mean()),
        "hv_composed_range": [float(min(p["hv_composed"] for p in points)),
                              float(max(p["hv_composed"] for p in points))],
        "hv_base_range": [float(min(p["hv_base"] for p in points)),
                          float(max(p["hv_base"] for p in points))],
    }

    for p in points:
        del p["_novel"]
        del p["_smis"]
        del p["_obj"]

    best = max(points, key=lambda p: p["top10"])
    out = {
        "reference": os.path.relpath(REFERENCE, REPO),
        "n_reference_molecules": len(ref_ms),
        "n_reference_scaffolds": len(ref_scaf),
        "points": sorted(points, key=lambda p: (p["operator"], p["w0"])),
        "adjacent_omega_jaccard": overlap,
        "hypervolume_paired": hv_paired,
        "hypervolume_pooled_by_operator": hv_pooled,
        "prior_chunk_control": control,
        "best_top10_over_omega": best["top10"],
        "best_top10_over_omega_at": "%s/w%.1f" % (best["operator"], best["w0"]),
        "uniform_top10": {p["operator"]: p["top10"] for p in points if abs(p["w0"] - 0.5) < 1e-9},
        "top10_base_range": [min(p["top10_base"] for p in points),
                             max(p["top10_base"] for p in points)],
        "nn_range": [min(p["nn_to_independent_prior"] for p in points),
                     max(p["nn_to_independent_prior"] for p in points)],
        "incomplete_points": missing,
    }

    dest = os.path.join(SWEEP, "summary.json")
    json.dump(out, open(dest, "w"), indent=2, sort_keys=True)

    print("reference: %s (%d molecules, %d scaffolds)"
          % (out["reference"], out["n_reference_molecules"], out["n_reference_scaffolds"]))
    print("\n%-9s %5s  %8s %8s  %8s  %8s  %6s  %6s | %8s %8s %8s"
          % ("operator", "w0", "c0", "c1", "top10", "NN(ind)", "nScaf", "novel",
             "HV comp", "HV base", "HV delta"))
    for p in out["points"]:
        print("%-9s %5.1f  %8.4f %8.4f  %8.4f  %8.4f  %6d  %5.2f | %8.5f %8.5f %+8.5f"
              % (p["operator"], p["w0"], p["c0"], p["c1"], p["top10"],
                 p["nn_to_independent_prior"], p["n_scaffolds"], p["novel_scaffold_frac"],
                 p["hv_composed"], p["hv_base"], p["hv_delta"]))
    print("\nhypervolume over (c0,c1), reference %s, one scale for every point:" % (REF,))
    print("  %d weight configurations produced %d distinct sampled populations"
          % (hv_paired["n_configurations"], hv_paired["n_distinct_populations"]))
    print("  composed range %.5f..%.5f   prior range %.5f..%.5f   (composed inside prior: %s)"
          % (*hv_paired["hv_composed_range"], *hv_paired["hv_base_range"],
             hv_paired["composed_range_inside_prior_range"]))
    print("  largest gain %+.5f, largest loss %+.5f, mean %+.5f"
          % (hv_paired["largest_gain"], hv_paired["largest_loss"], hv_paired["mean_delta"]))
    for op, v in hv_pooled.items():
        print("  %-9s union over omega %.5f vs best single omega %.5f  (ratio %.3f)"
              % (op, v["hv_union_over_omega"], v["hv_best_single_omega"],
                 v["union_over_best_single"]))
    if control:
        print("\nmatched control, %d unguided prior chunks of %d molecules:"
              % (control["n_chunks"], control["chunk_size"]))
        print("  NN to the same reference   %.4f  (range %.4f..%.4f)"
              % (control["nn_to_independent_prior_mean"],
                 control["nn_range"][0], control["nn_range"][1]))
        print("  novel-scaffold fraction    %.2f    (range %.2f..%.2f)"
              % (control["novel_scaffold_frac_mean"],
                 control["novel_scaffold_frac_range"][0],
                 control["novel_scaffold_frac_range"][1]))
    print("\noverlap between adjacent omega points (SAME sampling seed, so partly coupling):")
    for op, v in overlap.items():
        print("  %-9s mean Jaccard %.3f over %d adjacent pairs (%.3f..%.3f)"
              % (op, v["mean_jaccard"], v["n_pairs"], v["min_jaccard"], v["max_jaccard"]))
    print("\nbest top-10 over omega: %.4f at %s; prior top-10 range %.4f..%.4f"
          % (out["best_top10_over_omega"], out["best_top10_over_omega_at"],
             out["top10_base_range"][0], out["top10_base_range"][1]))
    if missing:
        print("\nincomplete points (%d): %s" % (len(missing), missing[:6]))
    print("\nwrote", os.path.relpath(dest, REPO))


if __name__ == "__main__":
    main()
