"""make_fig02_positional.py -- sampled-flip rate against sequence position,
pooled over configurations, one line per guide architecture.

Adapted from the parent repo's make_fig02_positional.py. Reads every
flip_report_*.json under results/molgpt/flips/ (06_flip_diagnostics.sh's
output) and averages `flip_rate_by_position` (t=1.0) across configs sharing
a guide architecture -- `null` entries (positions no trajectory in that
config reached) are excluded from the average at that position, not treated
as zero, matching flip_molgpt.py's own null-vs-zero discipline.
"""
import glob
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import FLIPS_DIR, GUIDE_COLOURS, savefig, require, plt

NAME_RE = re.compile(r"^sweep-[^-]+-(?P<guide>[^-]+)-")


def main():
    files = sorted(glob.glob(f"{FLIPS_DIR}/flip_report_*.json"))
    require(files[0] if files else FLIPS_DIR, "bash scripts_molgpt/06_flip_diagnostics.sh")

    by_guide = defaultdict(list)   # guide -> list of flip_rate_by_position lists
    for f in files:
        report = json.load(open(f))
        m = NAME_RE.match(report["label"])
        if not m:
            continue
        t1 = report.get("results_by_temp", {}).get("t1.0", {})
        pos = t1.get("flip_rate_by_position")
        if pos:
            by_guide[m.group("guide")].append(pos)

    fig, ax = plt.subplots(figsize=(6, 4))
    for guide, series_list in sorted(by_guide.items()):
        n_pos = max(len(s) for s in series_list)
        means = []
        for i in range(n_pos):
            vals = [s[i] for s in series_list if i < len(s) and s[i] is not None]
            means.append(sum(vals) / len(vals) if vals else None)
        xs = [i for i, v in enumerate(means) if v is not None]
        ys = [v for v in means if v is not None]
        ax.plot(xs, ys, color=GUIDE_COLOURS.get(guide, "#333333"), label=guide, marker=".")

    ax.set_xlabel("sequence position")
    ax.set_ylabel("coupled sampled-flip rate")
    ax.set_title("MolGPT")
    ax.legend()
    savefig(fig, "fig02_positional.png")


if __name__ == "__main__":
    main()
