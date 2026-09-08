"""make_fig12_mpo_components.py -- Figure 12 analogue: per-component scores
of the GuacaMol MPO objectives (osim/peri/fexo), one panel per benchmark.

Quetzal's own fig12 reads a pre-computed `extended.components` block from an
oracle-harvest JSON this pilot never produced. There is no equivalent
artifact here, so this script recomputes the same quantity directly: each
GuacaMol MPO benchmark's `objective` is a `GeometricMeanScoringFunction`
over a small number of leaf `scoring_functions` (2-4 per benchmark, e.g.
Tanimoto-FCFP4 + Tanimoto-ECFP6 + Rdkit-tpsa + Rdkit-logP for osimertinib),
and calling each leaf's own `.score(smiles)` gives the per-component value
the aggregate would otherwise hide. Computed here over the top-100 (by
log-reward) valid molecules of every config's `dumps/*/guided_smiles.txt`
-- same top-100 convention as `final_dump_molgpt.py`'s own `log_reward_top100`.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figstyle_pilot import rel, GUIDE_COLOURS, savefig, require, plt

sys.path.insert(0, rel("..", "molgpt_gfn"))

import scipy
if not hasattr(scipy, "histogram"):
    scipy.histogram = np.histogram
from rdkit import Chem

DUMPS_DIR = rel("results", "molgpt", "dumps")
NAME_RE = re.compile(r"^sweep-(?P<reward>[^-]+)-(?P<guide>[^-]+)-(?P<objective>[^-]+)-")
BENCH_FN = {"osim": "hard_osimertinib", "peri": "perindopril_rings", "fexo": "hard_fexofenadine"}


def component_label(sf):
    extra = getattr(sf, "fp_type", None)
    if extra is not None:
        return f"Tanimoto {extra}"
    desc = getattr(sf, "descriptor", None)
    return f"Rdkit {getattr(desc, '__name__', 'descriptor')}"


def load_objective(bench_key):
    from guacamol import standard_benchmarks as sb
    bench = getattr(sb, BENCH_FN[bench_key])()
    return bench.objective


def top100_smiles(dump_dir):
    smi_path = os.path.join(dump_dir, "guided_smiles.txt")
    r_path = os.path.join(dump_dir, "guided_rewards.npy")
    if not (os.path.exists(smi_path) and os.path.exists(r_path)):
        return []
    smiles = [s for s in open(smi_path).read().splitlines() if s]
    rewards = np.load(r_path)
    if len(smiles) != len(rewards) or not smiles:
        return []
    order = np.argsort(rewards)[::-1][:100]
    return [smiles[i] for i in order]


def main():
    benches = [b for b in ("osim", "peri", "fexo")
               if glob.glob(f"{DUMPS_DIR}/sweep-{b}-*/guided_smiles.txt")]
    require(DUMPS_DIR if benches else None, "bash scripts_molgpt/07_final_dump.sh")

    fig, axes = plt.subplots(1, len(benches), figsize=(5.2 * len(benches), 3.6), squeeze=False)
    axes = axes[0]

    for ax, bench_key in zip(axes, benches):
        obj = load_objective(bench_key)
        comps = obj.scoring_functions
        labels = [component_label(sf) for sf in comps]
        x = np.arange(len(comps))

        dirs = sorted(glob.glob(f"{DUMPS_DIR}/sweep-{bench_key}-*"))
        for d in dirs:
            name = os.path.basename(d)
            m = NAME_RE.match(name)
            if not m:
                continue
            smiles = top100_smiles(d)
            if not smiles:
                continue
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            mols = [mm for mm in mols if mm is not None]
            if not mols:
                continue
            ys = []
            for sf in comps:
                vals = [sf.score(Chem.MolToSmiles(mm)) for mm in mols]
                ys.append(float(np.mean(vals)))
            style = ":" if m.group("objective") == "rtb" else "-"
            ax.plot(x, ys, style, marker="o", ms=3, lw=1.0, alpha=0.6,
                    color=GUIDE_COLOURS.get(m.group("guide"), "0.5"))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(bench_key)

    axes[0].set_ylabel("component score (mean, top-100)")
    handles = [plt.Line2D([], [], color=c, lw=1.4) for c in GUIDE_COLOURS.values()]
    handles += [plt.Line2D([], [], color="0.3", lw=1.2, ls="-"),
                plt.Line2D([], [], color="0.3", lw=1.2, ls=":")]
    axes[-1].legend(handles, [f"guide: {g}" for g in GUIDE_COLOURS] + ["db", "rtb"],
                    frameon=False, fontsize=7)
    # No suptitle: the caption carries the description.
    fig.tight_layout()
    savefig(fig, "fig12_mpo_components.png")


if __name__ == "__main__":
    main()
