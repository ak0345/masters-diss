"""make_cmp02b_positional_atomidx.py -- the genuinely-commensurate version
of make_cmp02_positional.py: flip rate against ATOM-DECISION INDEX (the
k-th atom decision made along the trajectory) rather than raw sequence
position.

make_cmp02_positional.py's raw-position x-axis is not a comparable unit
across models -- most visibly for G2PT, whose rigid `<boc> ATOM IDX <sepc>`
grammar makes two out of every three raw positions a structural, never-an-
atom-decision step, producing a sawtooth comb pattern rather than a smooth
decay (see decisions.md, "why did we add atom-decision-index bucketing").
`flip_g2pt.py`/`flip_molgpt.py` were extended (same day) to also record
`flip_rate_by_atom_index`/`state_count_by_atom_index`, bucketing ONLY the
atom-decision steps themselves by a per-trajectory running atom-count --
this is the field this figure reads. Quetzal needs no such field: its own
`flip_rate_by_position` already IS atom-indexed, since every position in a
3D point cloud is an atom placement by construction (see gflow.py -- no
interleaved structural tokens the way MolGPT's SMILES or G2PT's graph
grammar have).

This does not replace make_cmp02_positional.py -- the raw-position version
stays, since the comb pattern it shows for G2PT is itself a real, legible
signature of that grammar worth keeping on record, not a rendering flaw to
hide. This figure is the one to read for "does the same qualitative decay
shape appear at a genuinely shared unit," which the raw-position figure
cannot answer for G2PT specifically.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_compare import (flip_reports, t1_block, savefig, require,
                               plt, MODEL_COLOURS, FLIPS_DIRS)

MAX_IDX = 60


def pooled_curve(model, max_idx):
    reports = flip_reports(model)
    flips = np.zeros(max_idx)
    states = np.zeros(max_idx)
    for label, report in reports.items():
        block = t1_block(model, report, regime="overall")
        if model == "Quetzal":
            # Already atom-indexed -- every raw position IS an atom decision.
            rate = block.get("flip_rate_by_position")
            counts = block.get("states_by_position")
        else:
            rate = block.get("flip_rate_by_atom_index")
            counts = block.get("state_count_by_atom_index")
        if not rate or not counts:
            continue
        n = min(max_idx, len(rate), len(counts))
        for i in range(n):
            c = counts[i] or 0
            r = rate[i]
            if c and r is not None:
                flips[i] += r * c
                states[i] += c
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(states > 0, flips / states, np.nan)
    return curve, states


def main():
    for model in ("Quetzal", "MolGPT", "G2PT"):
        require(FLIPS_DIRS[model], f"produce {model}'s flip_report_*.json first")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model in ("Quetzal", "MolGPT", "G2PT"):
        curve, states = pooled_curve(model, MAX_IDX)
        xs = [i for i, v in enumerate(curve) if not np.isnan(v)]
        ys = [curve[i] for i in xs]
        if not xs:
            print(f"[cmp02b] {model}: no atom-indexed data found -- rerun its flip diagnostics")
            continue
        ax.plot(xs, ys, "-", marker=".", lw=1.4, color=MODEL_COLOURS[model],
                label=f"{model} (n@idx0={int(states[0])})")

    ax.set_xlabel("atom-decision index (the k-th atom placed -- commensurate across models)")
    ax.set_ylabel("coupled sampled-flip rate (pooled across all configs)")
    ax.set_ylim(bottom=0)
    ax.set_title("By atom decision")
    ax.legend(frameon=False)
    savefig(fig, "cmp02b_positional_atomidx.png")


if __name__ == "__main__":
    main()
