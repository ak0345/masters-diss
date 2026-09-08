"""make_cmp07_bestofn_baseline.py -- baseline #1 from the ICLR-gap discussion:
is a trained guide doing anything a bigger unguided sample from the SAME
frozen prior wouldn't already give you for free?

Two things, both computed entirely from data already on disk (every
per-molecule log-reward from 07_final_dump.sh's own guided_rewards.npy /
base_rewards.npy -- no new sampling, no new training):

  1. Bootstrap 95% CIs on the top-10-of-N mean score, for both the base
     (unguided) and guided pools, per model per reward -- directly answers
     the adversarial review's W1 ("what is the bootstrap 95% interval on
     the frozen prior's top-10 score... does any claim survive?"). Printed,
     not just plotted, since this is the number that actually matters.

  2. A best-of-N harvest curve: repeatedly subsample the pooled UNGUIDED
     base rewards at increasing sample budgets N and average the top-10
     score, then mark each model's own best guided-config top-10 (the
     number the manuscript would call its headline result) as a horizontal
     line against that curve. Where the line crosses the curve is the
     "effective N" -- how many free unguided samples would already match
     the guide's best result. A guide result that never rises above the
     harvest curve's ceiling (its N=pool-size value) is indistinguishable
     from more free samples; one that clears the ceiling is real signal a
     bigger unguided draw could not have produced.

Base pools are reward-agnostic for MolGPT/G2PT (base generation doesn't
depend on which reward a config targets) but reward-specific for Quetzal
(its own `_base/<reward>/seed{0,42}` split) -- same convention as every
other comparison_figures script that reads raw dumps.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, savefig, require, plt, load_master,
                               MODEL_COLOURS, COMMON_REWARDS, REWARD_TITLE,
                               DUMPS_DIRS)

SEED = 0
N_BOOTSTRAP = 2000
HARVEST_NS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

QUETZAL_NAME_RE = re.compile(
    r"^sweep-(?P<reward>[a-z]+)-(?P<guide>base|hidden)-(?P<objective>db|rtb)"
    r"-replay_(?P<replay>on|off)-b(?P<beta>\d+)(?:-s\d+)?$")


def pooled_rewards(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            arr = np.load(p)
            if arr.size:
                out.append(np.exp(arr))  # stored as log-reward for all three legs
    return np.concatenate(out) if out else np.array([])


def quetzal_pools():
    base, guided = {}, {r: [] for r in COMMON_REWARDS}
    for r in COMMON_REWARDS:
        base[r] = pooled_rewards(
            glob.glob(f"{rel('results','dumps','_base')}/{r}/seed*/base_rewards.npy"))
    for d in sorted(glob.glob(f"{DUMPS_DIRS['Quetzal']}/sweep-*")):
        m = QUETZAL_NAME_RE.match(os.path.basename(d))
        if not m or m.group("replay") != "off" or m.group("beta") != "10":
            continue
        reward = m.group("reward")
        if reward not in COMMON_REWARDS:
            continue
        f = os.path.join(d, "seed0", "guided_rewards.npy")
        if os.path.exists(f):
            guided[reward].append(np.exp(np.load(f)))
    guided = {r: (np.concatenate(v) if v else np.array([])) for r, v in guided.items()}
    return base, guided


def pilot_pools(model):
    """Unlike base SMILES (reward-agnostic -- the frozen prior's own
    unconditional generation doesn't depend on which reward a config
    targets), base REWARDS are NOT reward-agnostic: each config's own
    base_rewards.npy scores that config's base molecules against THAT
    config's own reward function. Pooling base_rewards.npy across all
    configs regardless of reward mixes incompatible scales (e.g. nitrogen's
    easy near-1.0 scores contaminating osim/peri/fexo) -- caught before
    this ever reached a printed number: base top10 was landing at an
    identical ~0.9999 for all four rewards, which is what that bug looks
    like. Base rewards must be pooled per-reward, exactly like guided."""
    base = {r: [] for r in COMMON_REWARDS}
    guided = {r: [] for r in COMMON_REWARDS}
    for d in sorted(glob.glob(f"{DUMPS_DIRS[model]}/sweep-*")):
        reward = os.path.basename(d).split("-")[1]
        if reward not in COMMON_REWARDS:
            continue
        bf = os.path.join(d, "base_rewards.npy")
        if os.path.exists(bf):
            base[reward].append(np.exp(np.load(bf)))
        gf = os.path.join(d, "guided_rewards.npy")
        if os.path.exists(gf):
            guided[reward].append(np.exp(np.load(gf)))
    base = {r: (np.concatenate(v) if v else np.array([])) for r, v in base.items()}
    guided = {r: (np.concatenate(v) if v else np.array([])) for r, v in guided.items()}
    return base, guided


def top10_mean(x):
    if len(x) == 0:
        return np.nan
    k = min(10, len(x))
    return float(np.sort(x)[-k:].mean())


def bootstrap_top10_ci(x, rng, n_draw=None, n_boot=N_BOOTSTRAP):
    """95% percentile-bootstrap CI on the top-10-of-n_draw mean. n_draw
    defaults to len(x) (resample the pool at its own size)."""
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    n_draw = n_draw or len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(x, size=min(n_draw, len(x)), replace=True)
        boots[i] = top10_mean(s)
    return (float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def harvest_curve(x, rng, ns=HARVEST_NS, n_boot=200):
    out_n, out_mean = [], []
    for n in ns:
        if n > len(x):
            continue
        vals = [top10_mean(rng.choice(x, size=n, replace=(n > len(x)))) for _ in range(n_boot)]
        out_n.append(n)
        out_mean.append(float(np.mean(vals)))
    return out_n, out_mean


def best_guided_top10(model, reward):
    """The model's own best-of-48-configs guided top-10, matching what the
    manuscript's headline numbers are drawn from -- reused from master_table
    rather than recomputed, so this baseline is compared against the exact
    number the paper would report."""
    rows = load_master(model)
    if model == "Quetzal":
        cands = []
        for r in rows:
            if not r["name"].startswith("sweep-"):
                continue
            if r.get("reward") != reward or r.get("guide") not in ("base", "hidden") \
                    or r.get("objective") not in ("db", "rtb") or r.get("replay") != "off" \
                    or r.get("beta") != "10":
                continue
            v = r.get("guided_reward_top10_mean")
            if v not in (None, "", "None"):
                cands.append(float(v))
        return max(cands) if cands else np.nan
    else:
        cands = []
        for r in rows:
            if r.get("reward") != reward:
                continue
            v = r.get("guided_log_reward_top10_mean")
            if v not in (None, "", "None"):
                cands.append(np.exp(float(v)))
        return max(cands) if cands else np.nan


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(DUMPS_DIRS[model], f"produce {model}'s dumps first")

    rng = np.random.default_rng(SEED)
    pools = {"Quetzal": quetzal_pools(), "MolGPT": pilot_pools("MolGPT"), "G2PT": pilot_pools("G2PT")}

    print("=" * 78)
    print("Bootstrap 95% CIs on top-10-of-N mean score (N = pool's own size)")
    print("=" * 78)
    best_guided = {}
    for model, (base, guided) in pools.items():
        best_guided[model] = {}
        for r in COMMON_REWARDS:
            b_mean, b_lo, b_hi = bootstrap_top10_ci(base[r], rng)
            g_mean, g_lo, g_hi = bootstrap_top10_ci(guided[r], rng)
            best = best_guided_top10(model, r)
            best_guided[model][r] = best
            overlap = "OVERLAP" if (b_lo <= g_hi and g_lo <= b_hi) else "separated"
            print(f"[{model:8s}/{r:8s}] base n={len(base[r]):6d} top10={b_mean:.4f} "
                  f"[{b_lo:.4f},{b_hi:.4f}]  |  guided(pooled) n={len(guided[r]):6d} "
                  f"top10={g_mean:.4f} [{g_lo:.4f},{g_hi:.4f}]  ({overlap})  |  "
                  f"best single config top10={best:.4f}")

    fig, axes = plt.subplots(1, len(COMMON_REWARDS), figsize=(4.5 * len(COMMON_REWARDS), 4.3), squeeze=False)
    axes = axes[0]
    for ax, reward in zip(axes, COMMON_REWARDS):
        for model in ("Quetzal", "MolGPT", "G2PT"):
            base, _ = pools[model]
            ns, means = harvest_curve(base[reward], rng)
            if not ns:
                continue
            ax.plot(ns, means, color=MODEL_COLOURS[model], lw=1.6, marker=".", label=f"{model} (unguided harvest)")
            best = best_guided[model][reward]
            if best == best:  # not nan
                ax.axhline(best, color=MODEL_COLOURS[model], ls="--", lw=1.2, alpha=0.8,
                           label=f"{model} best guided top-10 ({best:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("N unguided samples drawn")
        ax.set_title(REWARD_TITLE.get(reward, reward), fontsize=9)
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel("top-10 mean score")
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp07_bestofn_baseline.png")


if __name__ == "__main__":
    main()
