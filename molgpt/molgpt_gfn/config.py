"""GFNConfig, adapted from gflow.py's `GFNConfig` for the MolGPT prior.

Every field below is generic (autoregressive-token / GFlowNet bookkeeping,
kept as-is) unless a comment says otherwise. Dropped relative to the parent
dataclass, all Quetzal/coordinate-specific:

  - `diff_steps`       coordinate-diffusion step count -- no coordinates here.
  - `mask_atoms`        atomic-number-vocab masking scheme -- MolGPT's vocab is
                        SMILES tokens, not atomic numbers; there is no
                        equivalent invalid-token set to mask (every one of the
                        94 tokens is a legal thing to emit at any position; the
                        model just may emit `<` early, which is a normal early
                        stop, not an invalid token).
  - `dataset`           selected an EDM atom-stability reference set; not used
                        by any reward kind in this pilot's scope (guacamol,
                        nitrogen_count).
  - `force_method`      selects an RLPF/xtb force reward -- geometry-only,
                        out of scope.
  - `use_ema_prior`      Quetzal's frozen prior has an EMA-vs-raw choice from
                        its own Lightning training; the vendored MolGPT
                        checkpoint is a single `state_dict`, no EMA variant.

Renamed: `quetzal_ckpt`/`train_module` -> `molgpt_ckpt` (the vendored GPT class
is imported directly by `molgpt_prior.py`, so there's no train_module
indirection to configure).

`vocab_size`/`max_len` defaults changed to MolGPT's geom-drugs-unconditional
values (39, 174, retrained 2026-09-04 -- see ../../shared_data/PROVENANCE.md)
instead of Quetzal's (128, 192) -- see `../vendor/PROVENANCE.md`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GFNConfig:
    molgpt_ckpt: str = "checkpoints/molgpt_geom_drugs_unconditional.pt"

    name: str = "molgpt-gfn-nitrogen--db-hidden-b10"
    devices: int = 1
    num_nodes: int = 1
    # Training seed: seeds guide initialisation, rollout sampling. Distinct
    # from a dump/eval-time reseed of an already-trained checkpoint.
    seed: int = 0
    debug: bool = False
    resume_path: str = None
    warm_start_guide: str = ""
    warm_start_source: str = "ema"

    bsz: int = 128
    max_len: int = 174          # MolGPT block_size (geom-drugs-unconditional, 2026-09-04)

    reward: str = "nitrogen_count"     # "nitrogen_count" | "guacamol"
    reward_beta: float = 10.0          # plan_molgpt.md pilot scope: beta=10 only
    invalid_logr: float = -5.0
    db_target_clip: float = -6.0
    reward_smiles: str = None          # guacamol kind: e.g. "hard_osimertinib"

    objective: str = "db"              # "db" | "rtb" -- pilot scope, no vargrad/kl
    logz_init: float = 0.0
    ratio_clip: float = 0.0
    # --- DB (detailed balance) flow head ---
    flow_hidden: int = 512
    flow_layers: int = 2
    db_interior_weight: float = 1.0

    vocab_size: int = 39         # geom_drugs_stoi.json (2026-09-04 GEOM-Drugs retrain)
    guide_hidden: int = 512
    guide_layers: int = 2
    # Pilot scope: hidden vs base (residual) only, no tempgain.
    use_hidden_guide: bool = True
    hidden_guide_out_residual: bool = True

    sample_temp: float = 2.0
    rand_eps: float = 0.2

    steps_per_epoch: int = 100
    max_epochs: int = 6
    lr: float = 1e-4
    logz_lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    wd: float = 0.0
    grad_clip: float = 1.0
    guide_ema_decay: float = 0.999

    # --- Replay buffer: out of scope for this pilot (plan_molgpt.md, "Replay:
    # off only"). Fields kept so the CLI surface matches scripts/01_train_guides.sh's,
    # but gflow_molgpt.py raises if use_replay is set True -- there is no
    # MolGPT-shaped replay buffer implementation yet (replay_buffer.py's
    # storage format is Quetzal atoms+coords specific; see plan_molgpt.md,
    # "Adapt" > "Replay buffer").
    use_replay: bool = False
    replay_capacity: int = 10000
    replay_strategy: str = "reward"
    replay_fraction: float = 0.25
    replay_warmup: int = 256
    replay_insert_valid_only: bool = True

    # Evaluation
    eval_n: int = 5000            # plan_molgpt.md: N=5,000, matching guide-sampling
                                   # convention, deliberately not the N=10,000 used
                                   # for the GEOM-Drugs reference in the 3D study.
    eval_base: bool = True
    topk_list: str = "1,10,100"

    fcd_ref_smiles: str = None
    fcd_enabled: bool = True

    final_n: int = 0
    final_dir: str = None

    num_workers: int = 4
    save_interval_minutes: int = 10

    # wandb, added 2026-09-04 (author instruction: "use wandb online") --
    # gflow.py logs every guide-training run to wandb (project="quetzal-gfn",
    # WandbLogger); this leg's Trainer had no wandb integration at all until
    # now. Mirrors gflow.py's project/entity convention, see fit()'s wandb.init.
    wandb: bool = True

    # ---- hang guard ----
    guard_reward_timeout: float = 20.0
    guard_stall_minutes: float = 30.0
    max_train_hours: float = 0.0
