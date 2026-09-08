"""make_cmp02_positional.py -- three-way comparison analogue of Quetzal's
Figure 2: coupled sampled-flip rate against sequence position, one pooled
line per MODEL (not per guide architecture, that split is each leg's own
fig02_positional.py) -- pooled from raw per-position flip/state counts
exactly as Quetzal's own figstyle.load_flip_reports does (numerators and
denominators summed across every config, divided once at the end, so a
40,000-state run isn't weighted the same as a 400-state one).

X-AXIS UNITS DIFFER BETWEEN MODELS AND ARE NOT MADE COMMENSURATE HERE.
Quetzal's position is an atom index in a 3D point cloud; MolGPT's is a
SMILES-token index (atoms interleaved with bracket/ring/bond syntax);
G2PT's is a token index in its own longer `<boc>...<eoc><bog>...<eog>`
grammar. Plotting all three on one shared x-axis shows whether the same
*qualitative* decay-then-flatten shape appears in every representation, not
a claim that position 5 means the same thing across models.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (flip_reports, t1_block, state_count_key, savefig, require,
                               plt, MODEL_COLOURS, FLIPS_DIRS)

MAX_POS = 60


def pooled_curve(model, max_pos):
    reports = flip_reports(model)
    flips = np.zeros(max_pos)
    states = np.zeros(max_pos)
    count_key = state_count_key(model)
    for label, report in reports.items():
        # "overall" regime for G2PT: the raw per-position signal pooled over
        # the whole sequence, same framing as fig02/fig06's own choice.
        block = t1_block(model, report, regime="overall")
        rate = block.get("flip_rate_by_position")
        counts = block.get(count_key)
        if not rate or not counts:
            continue
        n = min(max_pos, len(rate), len(counts))
        for i in range(n):
            c = counts[i] or 0
            r = rate[i]
            if c and r is not None:
                flips[i] += r * c
                states[i] += c
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(states > 0, flips / states, np.nan)
    return curve


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(FLIPS_DIRS[model], f"produce {model}'s flip_report_*.json first")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model in ("Quetzal", "MolGPT", "G2PT"):
        curve = pooled_curve(model, MAX_POS)
        xs = [i for i, v in enumerate(curve) if not np.isnan(v)]
        ys = [curve[i] for i in xs]
        if not xs:
            continue
        ax.plot(xs, ys, "-", marker=".", lw=1.4, color=MODEL_COLOURS[model], label=model)

    ax.set_xlabel("sequence position (units differ by model -- see docstring)")
    ax.set_ylabel("coupled sampled-flip rate (pooled across all configs)")
    ax.set_ylim(bottom=0)
    ax.set_title("By sequence position")
    ax.legend(frameon=False)
    savefig(fig, "cmp02_positional.png")


if __name__ == "__main__":
    main()
