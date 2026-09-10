#!/usr/bin/env python3
"""A quantitative pure-re-ranking null for the terminal top-10 gain.

Sections 3.3.1 and 3.3.3 argue, qualitatively, that the guides re-rank within chemistry
the prior already reaches rather than steer toward new chemistry. This script turns that
argument into a number to test the observed top-10 gain against.

THE NULL. A guide that changes nothing about the *set* of reachable molecules or their
rewards -- it only decides, per sample, whether to keep the frozen prior's own paired
draw (probability = identical_frac, already measured by analyse_chemistry.py under
paired sampling) or swap it for an INDEPENDENT draw from that same prior's own reward
pool (probability = 1 - identical_frac). No new chemistry, no reward shift: pure
re-ranking within what the prior already produces, at the measured rate the guide
actually leaves molecules unchanged.

METHOD. For each reward, load the frozen prior's own linear reward pool (from
base_rewards.npy, exponentiated exactly as aggregate_dumps.py does), build many
synthetic "null-guided" pools under the mixture above, and take their top-10 means.
That gives a null distribution for the top-10 GAIN a guide with this identical-fraction
and no other effect would produce. The observed gain is the hidden/DB cell in
tab_diss_landscape.tex, since that is the exact guide configuration
analyse_chemistry.py measured identical_frac on -- no other cell has a matching
identical-fraction to build a null from.

Run once per available prior dump seed (0 and 42) rather than pooling them: taking the
top-10 of the two seeds' pools concatenated is not the same statistic as averaging their
separate top-10s (concatenating inflates it, since a larger sample has more room for
extreme values), and Table 3.1's own reference is the latter.

USAGE
  python scripts/reranking_null.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REWARDS = ["osim", "peri", "fexo", "nitrogen"]
N_BOOTSTRAP = 5000
RNG_SEED = 0

# tab_diss_landscape.tex's hidden/DB cell and single prior figure, per reward --
# the same guide configuration and the same prior reference the null is tested against.
OBSERVED = {
    "osim":     {"guided": 0.7980, "prior": 0.7919},
    "peri":     {"guided": 0.4523, "prior": 0.4465},
    "fexo":     {"guided": 0.7126, "prior": 0.7078},
    "nitrogen": {"guided": 0.8668, "prior": 0.4356},
}


def load_prior_rewards(reward, seed):
    p = os.path.join(ROOT, "results", "dumps", "_base", reward, f"seed{seed}",
                      "base_rewards.npy")
    return np.exp(np.load(p))


def top10_mean(x):
    return np.sort(x)[-10:].mean()


def main():
    chem = json.load(open(os.path.join(
        ROOT, "results", "ablations", "chemistry", "summary.json")))
    rng = np.random.default_rng(RNG_SEED)

    out = {}
    for reward in REWARDS:
        p_unchanged = chem["quetzal"][reward]["identical_frac"]
        obs = OBSERVED[reward]
        observed_gain = obs["guided"] - obs["prior"]
        out[reward] = {"identical_frac": p_unchanged, "observed_gain": observed_gain,
                        "by_seed": {}}
        for seed in (0, 42):
            pool = load_prior_rewards(reward, seed)
            n = len(pool)
            prior_top10 = top10_mean(pool)

            null_top10s = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                keep = rng.random(n) < p_unchanged
                synth = pool.copy()
                n_swap = int(n - keep.sum())
                if n_swap > 0:
                    synth[~keep] = rng.choice(pool, size=n_swap, replace=True)
                null_top10s[i] = top10_mean(synth)
            null_gains = null_top10s - prior_top10
            null_mean, null_lo, null_hi = (float(null_gains.mean()),
                                            float(np.percentile(null_gains, 2.5)),
                                            float(np.percentile(null_gains, 97.5)))
            null_std = float(null_gains.std())
            z = (observed_gain - null_mean) / null_std if null_std > 0 else None

            out[reward]["by_seed"][seed] = {
                "n": n, "prior_top10_from_pool": prior_top10,
                "null_gain_mean": null_mean, "null_gain_ci95": [null_lo, null_hi],
                "z": z,
            }
            print(f"{reward:10s} seed{seed:<3d} p_unchg={p_unchanged:.4f} n={n} "
                  f"prior_top10(pool)={prior_top10:.4f} "
                  f"null_gain={null_mean:+.4f} [{null_lo:+.4f},{null_hi:+.4f}] "
                  f"observed_gain={observed_gain:+.4f} z={z:.2f}")

    out_path = os.path.join(HERE, "reranking_null_summary.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
