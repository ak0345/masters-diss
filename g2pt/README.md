# g2pt-gfn

The 2D-graph leg of the steerability extension — same guide/GFlowNet machinery as
`../molgpt/` (frozen prior, `HiddenGuide`/`LogitGuide`, DB/RTB training, flip diagnostics),
pointed at G2PT (Graph Generative Pre-trained Transformer, Tufts ML, ICML 2025) instead of
MolGPT. **This replaces an earlier, abandoned attempt at this leg using GraphINVENT** — see
`../papers/current/decisions.md`, the 2026-09-03 entries, for the full reasoning (GraphINVENT
had no isolable atom-type-only action head and no usable pretrained checkpoint; G2PT has both).

Self-contained and git-ignored from the parent `quetzal_gfn` repo, same as `molgpt/` — meant to
become its own repository.

## Status (2026-09-03)

**Code written and smoke-tested against the real pretrained checkpoint** (not a fake/random one
— unlike the MolGPT leg at the equivalent point, G2PT's checkpoint was usable from the start).
Confirmed so far, by actually running it, not assumed:

- `xchen16/g2pt-guacamol-small-deg` downloads and loads cleanly via
  `transformers.AutoModelForCausalLM`. Unconditional generation: **89.1% valid**, genuinely
  diverse drug-like SMILES.
- The sequence grammar is fully deterministic (`<boc> ATOM_t IDX_i <sepc> ... <eoc> <bog>
  IDX_a IDX_b BOND_t <sepg> ... <eog>`), confirmed by reading `vendor/G2PT/datasets_utils.py`
  directly — this is what makes Option A (node-type-only guidance, matching Quetzal/MolGPT's
  scope) genuinely implementable here, unlike GraphINVENT. See `vendor/PROVENANCE.md` and
  `papers/current/decisions.md`.
- A real (not fake-checkpoint) DB-objective training step ran end-to-end: guide construction,
  identity-at-init check, rollout, loss, backward, checkpoint save.

**Not yet resolved: an OOM at the real batch size (bsz=128), not yet cleanly diagnosed.** The
one throughput measurement attempted so far was contaminated by a concurrently-running MolGPT
sweep competing for the same GPU (both default to `cuda:0`) — see `vendor/PROVENANCE.md`. Redo
this measurement on a genuinely free GPU before trusting `scripts_g2pt/01_train_guides.sh`'s
`BSZ` default (currently a placeholder, `32`, chosen defensively pending real numbers, not
measured-safe). The likely cause, worth checking first: `max_len=614` (G2PT's block size) is
far longer than MolGPT's `100`, and this leg's rollout builder recomputes the full sequence from
scratch every step (`FrozenG2PT.hidden_states`, no KV-cache — matches `model.py`'s own
`generate()`, which also has none), so peak memory grows with sequence length in a way the
MolGPT leg's much shorter rollouts never stressed. Reducing `max_len` well below 614 (real
GuacaMol molecules don't need the model's absolute worst-case token budget) is a more promising
fix than just shrinking the batch, but hasn't been measured either — do the measurement before
picking a fix.

### What's confirmed (by reading source and running it) vs. still open

See `vendor/PROVENANCE.md` for the full detail. Headline: `HiddenGuide`/`LogitGuide` port
unmodified (G2PT's `lm_head` is a plain tied-embedding linear layer, structurally identical to
MolGPT's); the Option-A gate (`FrozenG2PT.is_atom_step`, applied in `gflow_g2pt.py`'s rollout
builder) is a purely syntactic, position-based rule verified against the actual tokenizer
grammar, not assumed from the paper. **Not yet verified**: that the gate actually produces
`delivered_frac` ≈ 1.0 at atom steps and *exactly* 0.0 at non-atom steps on a real trained guide
— `flip_g2pt.py` reports both regimes separately specifically so this can be checked (see its
module docstring), but no guide has finished training yet to check it against.

### What's next, in order

1. Resolve the OOM (see above) on a GPU not contended by another running sweep.
2. Once resolved, the equivalent of MolGPT's sequencing step 3–4: one config end-to-end
   (`python g2pt_gfn/gflow_g2pt.py --name smoke --reward nitrogen_count --objective db
   --max_epochs 1 --steps_per_epoch 20 --bsz <measured-safe>`), then
   `python g2pt_gfn/flip_g2pt.py --ckpt logs/g2pt-gfn/smoke/checkpoints/last.ckpt --label smoke
   --n_traj 200` — check `delivered_frac` reads ~1.0 at `atom_steps` and exactly 0.0 at
   `non_atom_steps` before trusting anything past this point.
3. `DRY=1 bash scripts_g2pt/01_train_guides.sh` to preview the 24-config grid, then run it.
4. `bash scripts_g2pt/06_flip_diagnostics.sh`.
5. Aggregate into `results/g2pt/master_table.csv`, same column-semantics discipline as the
   MolGPT leg's table (kept separate, not merged).
6. Composed guides only if time/budget remain, same caveat as the other two legs.

## Layout

```
g2pt_gfn/
  g2pt_prior.py     frozen G2PT wrapper: HF checkpoint load, hidden states, atom-step gating
  guides.py         HiddenGuide / LogitGuide / LogFlowHead -- identical copy, third relocation
  config.py         GFNConfig, adapted; adds atom_guide_only (the Option-A gate toggle)
  reward_adapter.py SMILES-native guacamol + nitrogen-fraction reward -- identical copy
  gflow_g2pt.py     training loop: atom-step-gated rollout builder + DB/RTB loss (ported verbatim)
  flip_g2pt.py      flip diagnostics, reported per-regime (atom_steps / non_atom_steps / overall)
scripts_g2pt/
  common.sh, 01_train_guides.sh, 06_flip_diagnostics.sh   mirror the other two legs' interface
vendor/G2PT/        tufts-ml/G2PT, vendored read-only (see vendor/PROVENANCE.md)
checkpoints/        empty -- G2PT has no local checkpoint file, it's resolved via HuggingFace
                    (transformers.AutoModelForCausalLM caches under ~/.cache/huggingface)
logs/g2pt-gfn/      run directories (git-ignored)
results/g2pt/       flip reports, master_table.csv (git-ignored except the aggregate table)
```

## The Option-A gate, in one paragraph

Unlike Quetzal (where atom-type IS the entire per-step action space) or GraphINVENT (where it
can't be isolated at all — see the superseded decision entry), G2PT emits atom type as one
category of token among several (node references, bond types, structural markers) in a single
flat vocabulary and a single softmax. What makes node-type-only guidance possible here is that
the *positions* where an atom-type token is legal are fully determined by the sequence grammar
(immediately after `<boc>` or `<sepc>`), so `gflow_g2pt.py`'s rollout builder can gate the guide
by decoding step: apply it only at atom-decision steps, substitute the untouched frozen-prior
logits everywhere else. This is enforced with `torch.where`, not by hoping the guide learns to
be a no-op elsewhere — see `_guided_step`'s docstring in `gflow_g2pt.py` for why that gives an
exact-zero gradient guarantee at non-atom steps, not an approximate one.
