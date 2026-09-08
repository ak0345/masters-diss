"""make_fig03_training_curves.py -- DB against RTB training curves, one panel
per reward, log-reward mean and validity fraction over training steps.

New relative to the parent repo's figure set (its make_fig18_training_curves.py
reads wandb logs directly; this pilot logs plain JSONL instead -- see
gflow_g2pt.py's `fit()`), but the comparison it draws is exactly the one
that surfaced the DB-vs-RTB asymmetry this pilot's training actually showed:
DB's training-time log-reward stays essentially flat at the invalid floor
while RTB's rises substantially, at a much higher validity fraction
throughout. Reads logs/g2pt-gfn/sweep-*/train.jsonl.
"""
import glob
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import LOGS_DIR, OBJECTIVE_COLOURS, REWARD_TITLE, savefig, require, plt

NAME_RE = re.compile(r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-")


def _smooth(xs, k=10):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    run_dirs = sorted(glob.glob(f"{LOGS_DIR}/sweep-*"))
    require(run_dirs[0] if run_dirs else LOGS_DIR, "bash scripts_g2pt/01_train_guides.sh")

    by_reward_obj = defaultdict(list)   # (reward, objective) -> list of per-step log_reward_mean series
    for d in run_dirs:
        name = d.rsplit("/", 1)[-1]
        m = NAME_RE.match(name)
        if not m:
            continue
        jsonl = f"{d}/train.jsonl"
        try:
            lines = [json.loads(l) for l in open(jsonl)]
        except FileNotFoundError:
            continue
        if not lines:
            continue
        series = [l["train/log_reward_mean"] for l in lines]
        by_reward_obj[(m.group("reward"), m.group("objective"))].append(series)

    rewards = sorted({r for r, _o in by_reward_obj})
    fig, axes = plt.subplots(1, len(rewards), figsize=(5 * len(rewards), 4), sharey=True)
    if len(rewards) == 1:
        axes = [axes]

    for ax, reward in zip(axes, rewards):
        for obj in ("db", "rtb"):
            series_list = by_reward_obj.get((reward, obj), [])
            if not series_list:
                continue
            n = min(len(s) for s in series_list)
            mean_series = [sum(s[i] for s in series_list) / len(series_list) for i in range(n)]
            ax.plot(_smooth(mean_series), color=OBJECTIVE_COLOURS.get(obj, "#333"),
                    label=f"{obj} (n={len(series_list)} seeds)")
        ax.set_title(REWARD_TITLE.get(reward, reward))
        ax.set_xlabel("training step")
    axes[0].set_ylabel("log-reward mean (10-step smoothed)")
    axes[0].legend()
    # No suptitle: the caption carries the description.
    savefig(fig, "fig03_training_curves.png")


if __name__ == "__main__":
    main()
