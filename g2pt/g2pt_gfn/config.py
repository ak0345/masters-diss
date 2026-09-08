"""GFNConfig, adapted from gflow.py's `GFNConfig` for the G2PT prior --
mirrors `molgpt/molgpt_gfn/config.py`'s adaptation reasoning throughout;
see that file's docstring for what's dropped and why (coordinate/diffusion
fields, atomic-number masking, EMA-prior selection -- none of that carries
over to a token-sequence transformer here either).

G2PT-specific differences from the MolGPT leg's config:

  - `vocab_size`/`max_len` defaults match G2PT's from-scratch GEOM-Drugs
    checkpoint (2026-09-04, padded vocab 128, block_size 700) instead of the
    old guacamol-small-deg checkpoint's (120, 614) -- see
    `../vendor/PROVENANCE.md` and `../../shared_data/PROVENANCE.md`. The
    real, unpadded vocab is 116 entries; `vocab_size` here is the tensor
    width (matches `FrozenG2PT.padded_vocab_size`), since that's what the
    guide's output layer must match.
  - `atom_guide_only: bool = True` is new -- this is what makes the G2PT
    leg's guide Option A (node-type-only, matching Quetzal/MolGPT) rather
    than Option B. When True, `gflow_g2pt.py`'s rollout builder applies the
    guide only at atom-decision steps (`FrozenG2PT.is_atom_step`) and uses
    the raw frozen-prior logits everywhere else; when False, the guide is
    applied at every step (available for a future ablation, not this
    pilot's grid default). See `papers/current/decisions.md`, "G2PT leg uses
    Option A."
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GFNConfig:
    model_name_or_path: str = "checkpoints/g2pt_geom_drugs_hf"

    name: str = "g2pt-gfn-nitrogen--db-hidden-b10"
    devices: int = 1
    num_nodes: int = 1
    seed: int = 0
    debug: bool = False
    resume_path: str = None
    warm_start_guide: str = ""
    warm_start_source: str = "ema"

    bsz: int = 128
    # 300, not the model's block_size (700): the guacamol-vocab checkpoint's
    # own max_len=300 was set by measuring real generated-completion lengths
    # against a no-KV-cache OOM ceiling (bsz=128 OOMs at max_len=614 on that
    # checkpoint). GEOM-Drugs' true encoded-sequence-length distribution over
    # the full 233,155-molecule training corpus (measured directly,
    # shared_data/PROVENANCE.md's molecule pool -- same BFS grammar, same
    # "3*n_nodes+4*n_edges+2" token cost per molecule) is p99=277, p999=316 --
    # 300 sits between those (~99.5% coverage) while staying well under half
    # of block_size=700, the same safety margin the guacamol checkpoint's
    # 300/614 ratio had. This dataclass default matches what
    # scripts_g2pt/01_train_guides.sh's own MAX_LEN default actually passes
    # to every sweep config via --max_len -- keep them in sync; see that
    # script's comment and vendor/PROVENANCE.md for the full reasoning.
    max_len: int = 300

    reward: str = "nitrogen_count"     # "nitrogen_count" | "guacamol"
    reward_beta: float = 10.0          # plan_molgpt.md pilot scope, mirrored here: beta=10 only
    invalid_logr: float = -5.0
    db_target_clip: float = -6.0
    reward_smiles: str = None          # guacamol kind: e.g. "hard_osimertinib"

    objective: str = "db"              # "db" | "rtb"
    logz_init: float = 0.0
    ratio_clip: float = 0.0
    flow_hidden: int = 512
    flow_layers: int = 2
    db_interior_weight: float = 1.0

    vocab_size: int = 128         # padded tensor width -- see module docstring
    guide_hidden: int = 512
    guide_layers: int = 2
    use_hidden_guide: bool = True
    hidden_guide_out_residual: bool = True
    atom_guide_only: bool = True  # Option A gate -- see module docstring

    sample_temp: float = 2.0
    rand_eps: float = 0.2

    steps_per_epoch: int = 100
    max_epochs: int = 5
    lr: float = 1e-4
    logz_lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    wd: float = 0.0
    grad_clip: float = 1.0
    guide_ema_decay: float = 0.999

    use_replay: bool = False
    replay_capacity: int = 10000
    replay_strategy: str = "reward"
    replay_fraction: float = 0.25
    replay_warmup: int = 256
    replay_insert_valid_only: bool = True

    eval_n: int = 5000
    eval_base: bool = True
    topk_list: str = "1,10,100"

    fcd_ref_smiles: str = None
    fcd_enabled: bool = True

    final_n: int = 0
    final_dir: str = None

    num_workers: int = 4
    save_interval_minutes: int = 10

    guard_reward_timeout: float = 20.0
    guard_stall_minutes: float = 30.0
    max_train_hours: float = 0.0

    # wandb, added 2026-09-04 (author instruction: "use wandb online") --
    # gflow.py logs every guide-training run to wandb (project="quetzal-gfn",
    # WandbLogger); this leg's Trainer had no wandb integration at all until
    # now. Mirrors gflow.py's project/entity convention, see fit()'s wandb.init.
    wandb: bool = True
