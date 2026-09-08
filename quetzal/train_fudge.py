"""train_fudge.py -- Quetzal mirror of molgpt/molgpt_gfn/train_fudge_molgpt.py
and g2pt/g2pt_gfn/train_fudge_g2pt.py: train a FUDGE-style future
discriminator using the exact same bare-LogitGuide architecture and
checkpoint format the DB/RTB "base" guides use, so it evaluates with the
UNCHANGED final_dump.py / flip diagnostics -- no separate FUDGE generation
code needed.

Built via `LitGFlowNet` itself (use_hidden_guide=False -> lit.guide is a
TempGainGuide wrapping a bare LogitGuide, `guide.guide` -- with
use_prior_temp=False and use_residual_gain=False, both GFNConfig's own
defaults, TempGainGuide's `guided_logits(prior_logits, h)` is IDENTICAL to
plain `prior_logits + guide(h)`, confirmed by LitGFlowNet's own init sanity
check -- so training only the inner `guide.guide` and leaving the (parameter-
less, since both heads are disabled) temp/gain wrapper alone reproduces bare
LogitGuide behaviour exactly, while keeping the checkpoint in the SAME
architecture and key-naming shape Quetzal's own "base" DB/RTB guides already
use). Trained by masked binary cross-entropy instead of any GFlowNet
objective: for each labeled (hidden_state, realized_atom_type,
eventual_success) triple from fudge_data.py, only the output dimension
corresponding to the atom type actually realized at that step is supervised.

Saved in the exact Lightning-checkpoint shape final_dump.py's own loader
expects (`ckpt["hyper_parameters"]["config"]` to rebuild `lit`, `ckpt["state_dict"]`
loaded with strict=False) -- see final_dump.py's checkpoint-loading block for
why this shape, not a bespoke one: it lets a FUDGE run be evaluated by
`final_dump.py --ckpt <this>` and the flip-diagnostics script completely
unchanged, exactly like a DB/RTB config.

Author instruction: same 5 epochs x 100 steps = 500 total gradient steps as
the two pilot legs' own FUDGE training (matching each other, not
necessarily Quetzal's own native guide sweep, which uses 6x100=600 --
see scripts/01_train_guides.sh -- a deliberate, disclosed simplification:
one uniform FUDGE training budget across the three-way comparison rather
than three slightly different ones). Same 3-hour training cap as every
other guide in this project (gflow.py's own --max_train_hours), even though
this loop is expected to finish in well under that.

Usage: python train_fudge.py --data fudge_data/osim.pt --name fudge-osim
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from gflow import LitGFlowNet, GFNConfig  # noqa: E402
from fudge_data import REWARD_FLAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="output of fudge_data.py")
    ap.add_argument("--name", required=True, help="checkpoint dir name, e.g. fudge-osim")
    ap.add_argument("--ckpt_root", default="logs/quetzal-gfn")
    ap.add_argument("--quetzal_ckpt", default="geom.ckpt")
    ap.add_argument("--guide_hidden", type=int, default=512)
    ap.add_argument("--guide_layers", type=int, default=2)
    ap.add_argument("--strength", type=float, default=1.0,
                     help="scales the trained weights before saving")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_train_hours", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    blob = torch.load(args.data, map_location="cpu")
    H, tok, y = blob["h"], blob["token"], blob["y"].float()
    print(f"[train_fudge] loaded {H.shape[0]} states, d_model={H.shape[1]}, "
          f"reward={blob['reward']}, success_rate={y.mean().item():.3f}")

    cfg = GFNConfig(name=args.name, quetzal_ckpt=args.quetzal_ckpt, objective="rtb",
                     use_hidden_guide=False, guide_hidden=args.guide_hidden,
                     guide_layers=args.guide_layers, seed=args.seed,
                     use_prior_temp=False, use_residual_gain=False,
                     **REWARD_FLAGS[blob["reward"]])
    lit = LitGFlowNet(asdict(cfg)).to(args.device)
    disc = lit.guide.guide  # the inner bare LogitGuide -- see module docstring
    opt = torch.optim.AdamW(disc.parameters(), lr=args.lr)

    n = H.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    H_train, tok_train, y_train = (H[train_idx].to(args.device), tok[train_idx].to(args.device),
                                    y[train_idx].to(args.device))
    H_val, tok_val, y_val = H[val_idx].to(args.device), tok[val_idx].to(args.device), y[val_idx].to(args.device)

    def save(epoch):
        sd = disc.state_dict()
        if args.strength != 1.0:
            last = f"net.{2 * (args.guide_layers - 1)}"
            sd = {k: (v * args.strength if k.startswith(last) else v) for k, v in sd.items()}
        disc.load_state_dict(sd)
        lit.guide_ema.module.guide.load_state_dict(disc.state_dict())
        ckpt_dir = Path(args.ckpt_root) / cfg.name / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"hyper_parameters": dict(lit.hparams), "state_dict": lit.state_dict(),
                    "epoch": epoch}, ckpt_dir / "last.ckpt")

    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        if time.time() - t0 > args.max_train_hours * 3600:
            print(f"[train_fudge] max_train_hours={args.max_train_hours} reached, stopping at epoch {epoch}")
            break
        disc.train()
        total_loss, total_n = 0.0, 0
        for _step in range(args.steps_per_epoch):
            idx = torch.randint(0, H_train.shape[0], (args.batch_size,), device=args.device)
            logits = disc(H_train[idx])
            chosen_logit = logits.gather(-1, tok_train[idx].unsqueeze(-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(chosen_logit, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.numel()
            total_n += idx.numel()
        train_loss = total_loss / total_n

        disc.eval()
        with torch.no_grad():
            logits = disc(H_val)
            chosen_logit = logits.gather(-1, tok_val.unsqueeze(-1)).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(chosen_logit, y_val).item()
            val_acc = ((chosen_logit > 0).float() == y_val).float().mean().item()
        print(f"[train_fudge] epoch {epoch} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save(epoch)

    ckpt_path = Path(args.ckpt_root) / cfg.name / "checkpoints" / "last.ckpt"
    print(f"[train_fudge] best val_loss={best_val_loss:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
