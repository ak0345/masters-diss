#!/usr/bin/env python3
"""make_fudge_table.py -- emits tab_fudge.tex.

The future-discriminator (FUDGE) baseline, one training seed per
architecture, read through each architecture's own unmodified pipeline:
dump_summary.json / summary.json for guided/base top-10, and the matching
flip report for delivered_frac, argmax_flip_rate and sample_flip_rate.

Reads the `_v2` (regression-target) dump/flip trees by default, not the
original `_v1` ones. The discriminator was originally trained as a binary
top-quartile classifier; checked post hoc against a majority-class floor,
it barely cleared or fell below that floor for all three architectures,
because a binary threshold discards nearly all the signal in a skewed
continuous reward. `train_fudge*.py --target r` retrains the identical
architecture as a regression onto each state's own standardised trajectory
log-reward instead, validated by held-out R^2/Pearson rather than accuracy;
every one of the 12 architecture-reward configs recovers positive held-out
signal under this target. `--v1` switches back to the original trees for
comparison, but the manuscript should cite the v2 numbers.

G2PT's flip numbers are read at the `atom_steps` regime, the actual analogue
of the other legs' single delivered_frac (see flip_g2pt.py's module
docstring) -- an earlier version of this table read `overall`, which pools
in the ~87% of steps the guide is gated off at by construction and reports
near-zero delivery and sample-flip regardless of what the guide does at the
atom-decision steps that matter. Fixed 2026-09-08, papers/current/decisions.md.

INPUTS (v2, default)
  results/fudge_dumps_v2/<reward>/dump_summary.json                  (Quetzal)
  molgpt/results/molgpt/fudge_dumps_v2/<reward>/summary.json
  g2pt/results/g2pt/fudge_dumps_v2/<reward>/summary.json
  results/fudge_flips_v2/flip_report_fudge-v2-<reward>.json              (Quetzal)
  molgpt/results/molgpt/fudge_flips_v2/flip_report_fudge-v2-<reward>.json
  g2pt/results/g2pt/fudge_flips_v2/flip_report_fudge-v2-<reward>.json

USAGE
  python scripts/make_fudge_table.py --out papers/current/src/tab_fudge.tex
"""
import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARDS = ["osim", "peri", "fexo", "nitrogen"]
REWARD_LABEL = {"osim": "Osimertinib MPO", "peri": "Perindopril MPO",
                "fexo": "Fexofenadine MPO", "nitrogen": "Nitrogen (control)"}


def models_for(v1):
    tag = "" if v1 else "_v2"
    ckpt_tag = "" if v1 else "v2-"
    return [
        ("Quetzal", f"results/fudge_dumps{tag}", f"results/fudge_flips{tag}", "flat", ckpt_tag),
        ("MolGPT", f"molgpt/results/molgpt/fudge_dumps{tag}",
         f"molgpt/results/molgpt/fudge_flips{tag}", "flat", ckpt_tag),
        ("G2PT", f"g2pt/results/g2pt/fudge_dumps{tag}",
         f"g2pt/results/g2pt/fudge_flips{tag}", "regime", ckpt_tag),
    ]


def load_dump(dump_dir, reward):
    for name in ("dump_summary.json", "summary.json"):
        path = os.path.join(REPO_ROOT, dump_dir, reward, name)
        if os.path.exists(path):
            return json.load(open(path))
    raise SystemExit(f"no dump summary under {dump_dir}/{reward}")


def flip_metrics(flips_dir, reward, schema, ckpt_tag):
    path = os.path.join(REPO_ROOT, flips_dir, f"flip_report_fudge-{ckpt_tag}{reward}.json")
    d = json.load(open(path))
    if schema == "flat":
        t1 = d.get("flip_temp1.0") or (d.get("results_by_temp") or {}).get("t1.0", {})
        return t1["delivered_frac"], t1["argmax_flip_rate"], t1["sample_flip_rate"]
    t1 = d["results_by_temp"]["t1.0"]
    block = t1["by_regime"]["atom_steps"]
    return block["delivered_frac"], block["argmax_flip_rate"], block["sample_flip_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "papers", "current", "src", "tab_fudge.tex"))
    ap.add_argument("--v1", action="store_true",
                     help="read the original classification-target trees instead of _v2")
    args = ap.parse_args()

    rows_out = []
    for model_label, dump_dir, flips_dir, schema, ckpt_tag in models_for(args.v1):
        for reward in REWARDS:
            dump = load_dump(dump_dir, reward)
            guided_top10 = dump["guided"]["log_reward_top10"]
            base_top10 = dump["base"]["log_reward_top10"]
            delivered, argmax_flip, sample_flip = flip_metrics(flips_dir, reward, schema, ckpt_tag)
            rows_out.append({
                "model": model_label, "reward": REWARD_LABEL[reward],
                "guided": guided_top10, "base": base_top10, "delivered": delivered,
                "argmax_flip": 100 * argmax_flip, "sample_flip": 100 * sample_flip,
            })

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption[The future-discriminator baseline]{The future-discriminator "
                 r"(FUDGE) baseline, one training seed per architecture, evaluated through "
                 r"each architecture's own unmodified pipeline, trained by regression onto "
                 r"each state's own trajectory log-reward (\Cref{sec:ch4-fudge-setup}). "
                 r"Top-10 is on the log-reward "
                 r"scale native to each master table (not comparable across architectures; "
                 r"comparable within a row across guided/base). Quetzal's FUDGE dump uses "
                 r"$n{=}500$ generated molecules per condition, an order of magnitude below "
                 r"the $n{=}5000$ used for MolGPT and G2PT and for every DB/RTB dump, because "
                 r"of the coordinate-diffusion cost noted in \Cref{sec:fudge}. Delivered and "
                 r"flip read G2PT's \texttt{atom\_steps} regime, the actual analogue of the "
                 r"other two architectures' single delivered fraction.}")
    lines.append(r"\label{tab:fudge}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Reward & Guided top-10 & Base top-10 & Delivered & "
                 r"Argmax flip \% & Sample flip \% \\")
    lines.append(r"\midrule")
    for row in rows_out:
        lines.append(
            f"{row['model']} & {row['reward']} & {row['guided']:.4f} & {row['base']:.4f} & "
            f"{row['delivered']:.3f} & {row['argmax_flip']:.3f} & {row['sample_flip']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
