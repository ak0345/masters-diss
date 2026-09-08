"""train_fudge_molgpt.py -- train a FUDGE-style future-discriminator using
the exact same LogitGuide architecture and checkpoint format the DB/RTB
guides use, so it can be evaluated with the UNCHANGED final_dump_molgpt.py/
flip_molgpt.py -- no separate FUDGE generation code needed at all.

Built via Trainer itself (use_hidden_guide=False -> Trainer.guide is a
LogitGuide, h -> vocab_size, zero-init last layer -- the same "starts as a
no-op" discipline the DB/RTB guides use), then trained by masked binary
cross-entropy instead of any GFlowNet objective: for each labeled
(hidden_state, realized_token, eventual_success) triple from
fudge_data_molgpt.py, only the output dimension corresponding to the token
actually realized at that step is supervised. The other vocab dimensions
are simply not evaluated on that example; averaged over many states where
each atom token was the one realized, this is a standard sparse/contextual-
bandit-style training setup, not a special case.

At generation time, `prior_logits + strength * guide(h)` (strength folded
into the saved weights, see --strength) IS exactly LogitGuide's own
`guided_logits_dispatch` -- meaning `final_dump_molgpt.py --ckpt <this
checkpoint>` and `flip_molgpt.py --ckpt <this checkpoint>` already work
unchanged, evaluating FUDGE exactly like any DB/RTB config.

Trainer's own __init__ still builds a flow_head or logZ depending on
cfg.objective (unused here -- objective='rtb' picked only because it is the
cheaper of the two to construct, a scalar rather than a network) and runs
its own guide-identity-at-init sanity check, since this reuses Trainer
wholesale rather than reimplementing guide construction from scratch.

Usage: python train_fudge_molgpt.py --data results/molgpt/fudge_data/osim.pt \
         --name fudge-osim
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GFNConfig  # noqa: E402
from gflow_molgpt import Trainer  # noqa: E402
from fudge_data_molgpt import REWARD_FLAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="output of fudge_data_molgpt.py")
    ap.add_argument("--name", required=True, help="checkpoint dir name, e.g. fudge-osim")
    ap.add_argument("--ckpt_root", default="logs/molgpt-gfn")
    ap.add_argument("--molgpt_ckpt", default="checkpoints/molgpt_geom_drugs_unconditional.pt")
    ap.add_argument("--guide_hidden", type=int, default=512)
    ap.add_argument("--guide_layers", type=int, default=2)
    ap.add_argument("--strength", type=float, default=1.0,
                     help="scales the trained weights before saving, so the "
                          "saved checkpoint needs no separate inference-time knob")
    ap.add_argument("--lr", type=float, default=1e-3)
    # Author instruction: same 5 epochs x 100 steps = 500 total gradient
    # steps as the real DB/RTB guide sweep (scripts_molgpt/01_train_guides.sh's
    # MAX_EPOCHS=5, STEPS=100) -- a matched training BUDGET, not just a
    # matched wall-clock cap, so FUDGE isn't trained longer just because
    # each step here is cheap. Steps are random mini-batches (with
    # replacement across epochs), not fixed shuffled passes over the
    # dataset -- the direct analogue of "epoch" here since DB/RTB's own
    # epochs are on-policy rollout batches, not dataset passes either.
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--val_frac", type=float, default=0.1)
    # Author instruction: same 3-hour training cap as the real DB/RTB guides
    # (see gflow.py/rtb_finetune.py's own --max_train_hours) applied uniformly
    # across every model's guide training, FUDGE included, even though this
    # loop is expected to finish in well under that (plain BCE on already-
    # collected data, no rollout during training) -- a safety cap, not a
    # budget expected to bind.
    ap.add_argument("--max_train_hours", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    blob = torch.load(args.data, map_location="cpu")
    H, tok, y = blob["h"], blob["token"], blob["y"].float()
    print(f"[train_fudge] loaded {H.shape[0]} states, d_model={H.shape[1]}, "
          f"reward={blob['reward']}, success_rate={y.mean().item():.3f}")

    cfg = GFNConfig(name=args.name, molgpt_ckpt=args.molgpt_ckpt, objective="rtb",
                     use_hidden_guide=False, guide_hidden=args.guide_hidden,
                     guide_layers=args.guide_layers, seed=args.seed,
                     **REWARD_FLAGS[blob["reward"]])
    trainer = Trainer(cfg, device=args.device, ckpt_root=args.ckpt_root)
    disc = trainer.guide  # LogitGuide: h -> vocab_size, zero-init last layer
    opt = torch.optim.AdamW(disc.parameters(), lr=args.lr)

    n = H.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    H_train, tok_train, y_train = (H[train_idx].to(args.device), tok[train_idx].to(args.device),
                                    y[train_idx].to(args.device))
    H_val, tok_val, y_val = H[val_idx].to(args.device), tok[val_idx].to(args.device), y[val_idx].to(args.device)

    def save(val_loss, val_acc, epoch):
        # Fold --strength into the saved weights (scale the zero-init'd last
        # layer's weight+bias) so final_dump_molgpt.py needs no FUDGE-aware
        # inference-time knob -- prior_logits + guide(h) is already the
        # correctly-scaled guided logits.
        sd = {k: (v * args.strength if "net" in k and ("weight" in k or "bias" in k)
                  and k.startswith(f"net.{2 * (args.guide_layers - 1)}") else v)
              for k, v in disc.state_dict().items()}
        trainer.guide.load_state_dict(sd)
        trainer.guide_ema.module.load_state_dict(sd)
        trainer.save_checkpoint(epoch=epoch)
        trainer.guide.load_state_dict(disc.state_dict())  # restore unscaled for continued training

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
            save(val_loss, val_acc, epoch)

    ckpt_path = Path(args.ckpt_root) / cfg.name / "checkpoints" / "last.ckpt"
    print(f"[train_fudge] best val_loss={best_val_loss:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
