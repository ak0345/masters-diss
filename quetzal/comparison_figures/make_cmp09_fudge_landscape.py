"""make_cmp09_fudge_landscape.py -- cross-model comparison of guided top-10
score under the two guide training objectives (DB/RTB vs. FUDGE), one panel
per common reward. Companion to cmp08 (which compares flip-diagnostic
metrics the same way) and cmp01 (which compares DB/RTB alone against
published/GEOM baselines) -- see both for the fuller context this one
panel focuses down from.

DB/RTB scores reused directly from make_cmp01_landscape.py's own row
readers (best-of-48-configs, matching that figure's own convention).
FUDGE scores read from each leg's fudge_dumps/<reward>/{summary,dump_summary}.json
-- one guide per reward, no sweep to take a "best of" over.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, savefig, require, plt,
                               MODEL_COLOURS, COMMON_REWARDS, REWARD_TITLE)
from make_cmp01_landscape import quetzal_rows, pilot_rows

FUDGE_DUMPS = {
    "Quetzal": (rel("results", "fudge_dumps"), "dump_summary.json"),
    "MolGPT": (rel("molgpt", "results", "molgpt", "fudge_dumps"), "summary.json"),
    "G2PT": (rel("g2pt", "results", "g2pt", "fudge_dumps"), "summary.json"),
}


def fudge_score(model, reward):
    import os
    root, fname = FUDGE_DUMPS[model]
    path = os.path.join(root, reward, fname)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    v = d.get("guided", {}).get("log_reward_top10")
    return math.exp(v) if v is not None else None


def dbrtb_best(model, data_fn, leg_dir=None):
    rows = data_fn() if leg_dir is None else data_fn(model, leg_dir)
    best = {}
    for r in rows:
        rw = r["reward"]
        if rw not in best or r["score"] > best[rw]:
            best[rw] = r["score"]
    return best


def main():
    for model, (root, _fname) in FUDGE_DUMPS.items():
        require(root, f"produce {model}'s FUDGE dumps first")

    dbrtb = {
        "Quetzal": dbrtb_best("Quetzal", quetzal_rows),
        "MolGPT": dbrtb_best("MolGPT", pilot_rows, "molgpt"),
        "G2PT": dbrtb_best("G2PT", pilot_rows, "g2pt"),
    }
    fudge = {m: {r: fudge_score(m, r) for r in COMMON_REWARDS} for m in FUDGE_DUMPS}

    for m in FUDGE_DUMPS:
        print(f"[{m}] DB/RTB best", {r: round(v, 4) for r, v in dbrtb[m].items()})
        print(f"[{m}] FUDGE     ", {r: round(v, 4) if v is not None else None for r, v in fudge[m].items()})

    models = ["Quetzal", "MolGPT", "G2PT"]
    fig, axes = plt.subplots(1, len(COMMON_REWARDS), figsize=(4.2 * len(COMMON_REWARDS), 4.2))
    for ax, reward in zip(axes, COMMON_REWARDS):
        x = np.arange(len(models))
        width = 0.35
        db_vals = [dbrtb[m].get(reward, np.nan) for m in models]
        fu_vals = [fudge[m].get(reward) or np.nan for m in models]
        ax.bar(x - width / 2, db_vals, width, color=[MODEL_COLOURS[m] for m in models],
               alpha=0.45, label="DB/RTB (best)")
        ax.bar(x + width / 2, fu_vals, width, color=[MODEL_COLOURS[m] for m in models],
               alpha=0.95, label="FUDGE")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(REWARD_TITLE.get(reward, reward), fontsize=9)
        ax.set_ylim(0, 1.05 if reward != "nitrogen" else None)

    axes[0].set_ylabel("top-10 mean score")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.45, label="DB/RTB (best of grid)"),
               plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.95, label="FUDGE")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp09_fudge_landscape.png")


if __name__ == "__main__":
    main()
