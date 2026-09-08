#!/usr/bin/env python3
"""
make_fig19_beta100_training.py -- Figure 19: training at beta=100 does not
converge the way it does at beta=10, for either loss.

Figure 18 shows DB against RTB at beta=10 for one representative config. This
figure holds the config fixed (hidden guide, Osimertinib, replay off, seed 0)
and instead compares beta=10 against beta=100, for both training objectives,
to check whether the validity decline already reported for beta=100
(Section~\\ref{sec:training_objectives}, abstract) shows up in training itself
rather than only in the terminal sample.

Panels:
  A  training loss, each objective's own scale (DB: db/terminal_loss;
     RTB: train/loss -- the key RTB runs actually log under, not rtb/loss)
  B  fraction of the training batch that converts to a valid molecule
     (train/valid_frac), over training

INPUTS
  logs/wandb/run-*/files/config.yaml   to find the run directory for each config
  logs/wandb/run-*/run-*.wandb         the local wandb datastore (read directly;
                                        no network access, no wandb server needed)

USAGE
  python figures/make_fig19_beta100_training.py --out out/fig19_beta100_training.pdf
"""
import argparse

import matplotlib.pyplot as plt

import figstyle as fs
from make_fig18_training_curves import find_run, series

PAIRS = [
    ("sweep-osim-hidden-db-replay_off-b10-s0", "DB, $\\beta=10$", "#4C72B0", "-"),
    ("sweep-osim-hidden-db-replay_off-b100-s0", "DB, $\\beta=100$", "#4C72B0", "--"),
    ("sweep-osim-hidden-rtb-replay_off-b10-s0", "RTB, $\\beta=10$", "#DD8452", "-"),
    ("sweep-osim-hidden-rtb-replay_off-b100-s0", "RTB, $\\beta=100$", "#DD8452", "--"),
]


def loss_series(name, rows):
    loss_key = "db/terminal_loss" if "-db-" in name else "train/loss"
    xs, ys = series(rows, loss_key)
    if not xs:
        candidates = {k for r in rows for k in r if k.endswith("loss")}
        for k in candidates:
            xs, ys = series(rows, k)
            if xs:
                loss_key = k
                break
    return xs, ys, loss_key


def main():
    ap = argparse.ArgumentParser()
    fs.add_arg_common(ap, "out/fig19_beta100_training.pdf")
    args = ap.parse_args()
    fs.use_paper_style()

    runs = {}
    for name, label, colour, ls in PAIRS:
        rows = find_run(name)
        if not rows:
            print(f"[fig19] no wandb run found for {name}")
            continue
        runs[name] = (rows, label, colour, ls)

    if not runs:
        print("[fig19] no runs found, nothing to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(args.width * 2 / 3, 2.8))

    ax = axes[0]
    for name, (rows, label, colour, ls) in runs.items():
        xs, ys, loss_key = loss_series(name, rows)
        ax.plot(xs, ys, color=colour, linestyle=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Training loss")
    ax.legend()

    ax = axes[1]
    for name, (rows, label, colour, ls) in runs.items():
        xs, ys = series(rows, "train/valid_frac")
        ax.plot(xs, ys, color=colour, linestyle=ls, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of batch valid")
    ax.set_title("Training-batch validity")
    ax.legend()

    fs.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()
