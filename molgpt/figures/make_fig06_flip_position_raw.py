"""make_fig06_flip_position_raw.py -- Figure 6 analogue: sampled-flip rate by
sequence position, individual configs (not pooled).

Adapted from the parent repo's make_fig06_flip_position_raw.py. Supports
fig02_positional.py, which pools over configurations sharing a guide
architecture -- this draws every individual flip_report_*.json as its own
faint line, colour-coded by guide, so the decay pattern can be seen to hold
config-by-config rather than being an artifact of averaging. Positions no
trajectory reached (`state_count_by_position[i] == 0`) are plotted as gaps,
not zero, matching flip_molgpt.py's own null-vs-zero discipline.
"""
import glob
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import FLIPS_DIR, GUIDE_COLOURS, savefig, require, plt

NAME_RE = re.compile(r"^sweep-[^-]+-(?P<guide>[^-]+)-")
MAX_POS = 40


def main():
    files = sorted(glob.glob(f"{FLIPS_DIR}/flip_report_*.json"))
    require(files[0] if files else FLIPS_DIR, "bash scripts_molgpt/06_flip_diagnostics.sh")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    curves = []
    for f in files:
        report = json.load(open(f))
        label = report["label"]
        m = NAME_RE.match(label)
        if not m:
            continue
        guide = m.group("guide")
        t1 = report.get("results_by_temp", {}).get("t1.0", {})
        rate = t1.get("flip_rate_by_position")
        counts = t1.get("state_count_by_position")
        if not rate:
            continue
        n = min(MAX_POS, len(rate))
        xs, ys = [], []
        for i in range(n):
            if counts and i < len(counts) and not counts[i]:
                continue
            if rate[i] is None:
                continue
            xs.append(i)
            ys.append(rate[i])
        if not xs:
            continue
        ax.plot(xs, ys, "-", lw=1.0, alpha=0.5,
                color=GUIDE_COLOURS.get(guide, "0.5"), zorder=2)
        curves.append((ys[0], label))

    if not curves:
        raise SystemExit("[FATAL] no usable per-position curves found")

    curves.sort(key=lambda t: -t[0])
    hi, lo = curves[0], curves[-1]
    print(f"[fig06] {len(curves)} configs | position-0 flip rate: "
          f"max {hi[0]:.4f} ({hi[1]}), min {lo[0]:.4f} ({lo[1]})")

    ax.set_xlabel("sequence position (atom decision)")
    ax.set_ylabel("coupled sampled-flip rate")
    ax.set_ylim(bottom=0)
    ax.set_title("MolGPT")
    handles = [plt.Line2D([], [], color=c, lw=1.4) for c in GUIDE_COLOURS.values()]
    ax.legend(handles, [f"guide: {g}" for g in GUIDE_COLOURS], frameon=False)
    savefig(fig, "fig06_flip_position_raw.png")


if __name__ == "__main__":
    main()
