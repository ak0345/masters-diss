"""make_cmp01_landscape.py -- three-way comparison analogue of Quetzal's own
Figure 1: where each model's guided configurations land on a single score
axis, against the shared GEOM-Drugs best-of-10k baseline and (where one
exists) the published GuacaMol baselines.

One panel per reward common to all three legs' pilot grids (nitrogen, osim,
peri, fexo). Quetzal's own master_table.csv already reports the raw [0,1]
score directly (`guided_reward_top10_mean`); MolGPT's and G2PT's final_dump
scripts only recorded the log-reward, so their raw score is recovered as
exp(log_reward_top10_mean) -- exact for every entry that contributed to a
top-10/top-100 mean, since `reward_adapter.build_reward_smiles` computes
`log(score)` from exactly this score and only valid molecules ever enter
that mean (see final_dump_molgpt.py/final_dump_g2pt.py's `dump()`).

Quetzal's rows are restricted to `sweep-*` (excluding `compose-*`
composition-ablation runs, which have no MolGPT/G2PT equivalent) and to
guide in {base,hidden}, objective in {db,rtb}, replay=off, beta=10 -- the
exact grid the pilots share with it.
"""
import csv
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (rel, load_master, savefig, require, plt, json,
                               MODEL_COLOURS, COMMON_REWARDS, REWARD_TITLE,
                               PUBLISHED, GEOM_BASELINE_PATH)

REWARD_NORM = {"osimertinib": "osim", "perindopril": "peri"}


def quetzal_rows():
    rows = load_master("Quetzal")
    out = []
    for r in rows:
        if not r["name"].startswith("sweep-"):
            continue
        reward = REWARD_NORM.get(r.get("reward"), r.get("reward"))
        if reward not in COMMON_REWARDS:
            continue
        if r.get("guide") not in ("base", "hidden") or r.get("objective") not in ("db", "rtb"):
            continue
        if r.get("replay") != "off" or r.get("beta") != "10":
            continue
        v = r.get("guided_reward_top10_mean")
        if not v:
            continue
        out.append({"reward": reward, "guide": r["guide"], "objective": r["objective"],
                    "score": float(v)})
    return out


def pilot_rows(model, leg_dir):
    rows = load_master(model)
    out = []
    for r in rows:
        reward = r.get("reward")
        if reward not in COMMON_REWARDS:
            continue
        v = r.get("guided_log_reward_top10_mean")
        if not v or v in ("", "None"):
            continue
        out.append({"reward": reward, "guide": r["guide"], "objective": r["objective"],
                    "score": math.exp(float(v))})
    return out


def main():
    require(rel("results", "dumps", "_aggregate", "master_table.csv"), "bash scripts/08_analysis.sh aggregate")
    require(rel("molgpt", "results", "molgpt", "_aggregate", "master_table.csv"),
            "bash molgpt/scripts_molgpt/08_aggregate.sh")
    require(rel("g2pt", "results", "g2pt", "_aggregate", "master_table.csv"),
            "bash g2pt/scripts_g2pt/08_aggregate.sh")

    data = {
        "Quetzal": quetzal_rows(),
        "MolGPT": pilot_rows("MolGPT", "molgpt"),
        "G2PT": pilot_rows("G2PT", "g2pt"),
    }
    geom_baseline = json.load(open(GEOM_BASELINE_PATH))

    fig, axes = plt.subplots(1, len(COMMON_REWARDS), figsize=(4.2 * len(COMMON_REWARDS), 4.2),
                              squeeze=False)
    axes = axes[0]
    models = ["Quetzal", "MolGPT", "G2PT"]

    for ax, reward in zip(axes, COMMON_REWARDS):
        for xi, model in enumerate(models):
            pts = [d["score"] for d in data[model] if d["reward"] == reward]
            if not pts:
                continue
            jitter = [xi + 0.08 * ((i % 5) - 2) for i in range(len(pts))]
            ax.scatter(jitter, pts, color=MODEL_COLOURS[model], alpha=0.75,
                       s=28, edgecolors="none", zorder=3)

        gb = geom_baseline.get(reward, {}).get("top10")
        if gb is not None:
            ax.axhline(gb, color="0.4", ls="--", lw=1.1, zorder=1,
                       label="GEOM-Drugs best-of-10k")
        for label, val in PUBLISHED.get(reward, {}).items():
            ax.axhline(val, color="0.6", ls=":", lw=1.0, zorder=1)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(REWARD_TITLE.get(reward, reward))
        ax.set_ylim(0, 1.05 if reward != "nitrogen" else None)

    axes[0].set_ylabel("top-10 mean score")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=7, label=m) for m, c in MODEL_COLOURS.items()]
    handles.append(plt.Line2D([0], [0], color="0.4", ls="--", lw=1.1, label="GEOM-Drugs best-of-10k"))
    handles.append(plt.Line2D([0], [0], color="0.6", ls=":", lw=1.0, label="published (Brown et al. 2019)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=8)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "cmp01_landscape.png")


if __name__ == "__main__":
    main()
