# vendor/G2PT provenance

Vendored from `github.com/tufts-ml/G2PT` (Graph Generative Pre-trained Transformer, ICML 2025,
`arXiv:2501.01073`), commit `e8957613d82ca5fdb916f6d63f76110d9431f5e9`, via a shallow clone on
2026-09-03. `.git` stripped after cloning (same reasoning as `molgpt/vendor/PROVENANCE.md`).
Replaces an earlier, abandoned attempt with `MolecularAI/GraphINVENT` — see
`papers/current/decisions.md`, 2026-09-03 entries, for why.

Kept read-only / unmodified. `g2pt_gfn/` imports from it only for reference (the model class
`model.py` defines is what the published HF checkpoints *are*, but this pilot loads them via
`transformers.AutoModelForCausalLM` directly, not this file's own `GPT` class — see
`g2pt_prior.py`'s docstring) and for the `guacamol` tokenizer's `vocab.json`.

## What's confirmed by reading this source (2026-09-03), not assumed

- **Architecture**: `model.py`'s `GPT` is nanoGPT (near-verbatim, same license lineage) — a
  plain decoder-only transformer, tied token embedding/head, single flat `lm_head =
  nn.Linear(n_embd, vocab_size, bias=False)` applied post-`ln_f`. Structurally identical to
  MolGPT's head, not GraphINVENT's fused multi-axis one.
- **Sequence grammar** (traced from `datasets_utils.py`'s `to_seq_by_bfs`/`to_seq_by_deg`
  encoders and `seq_to_mol` decoder):
  ```
  <boc> ATOM_t0 IDX_0 <sepc> ATOM_t1 IDX_1 <sepc> ... ATOM_tn IDX_n <eoc>
  <bog> IDX_a IDX_b BOND_t <sepg> IDX_c IDX_d BOND_t <sepg> ... <eog>
  ```
  Fully deterministic and position-typed: an atom-type token is legal, and only legal,
  immediately after `<boc>` or `<sepc>`. This is what makes Option-A-style node-type-only
  guidance implementable *in the rollout builder*, with zero changes to the frozen model — see
  `papers/current/decisions.md`, "G2PT leg uses Option A."
- **Vocabulary** (`tokenizers/guacamol/vocab.json`, 112 real entries): 6 special/structural
  tokens (`<boc> <eoc> <sepc> <bog> <eog> <sepg>`), `IDX_0`..`IDX_89` (90 node-reference
  tokens), `ATOM_C ATOM_N ATOM_O ATOM_F ATOM_B ATOM_Br ATOM_Cl ATOM_I ATOM_P ATOM_S ATOM_Se
  ATOM_Si` (12 atom types, one contiguous index range, indices 96–107), `BOND_SINGLE
  BOND_DOUBLE BOND_TRIPLE BOND_AROMATIC` (4 bond types, indices 108–111).
- **Checkpoint**: `xchen16/g2pt-guacamol-small-deg` on HuggingFace is real and loads cleanly —
  confirmed by actually downloading and running it (not assumed from the model card), 2026-09-03.
  `AutoConfig`: `n_embd=384, n_head=6, n_layer=6, n_positions=614`. Published as a standard
  `transformers.GPT2LMHeadModel` (`model.py`'s `to_hf()` is what produces this format from the
  repo's own nanoGPT-style class), so `g2pt_prior.py` loads it via `AutoModelForCausalLM`
  directly rather than reimplementing the forward pass by hand the way `molgpt_prior.py` had to.
  Unconditional sampling at temperature 1.0: **89.1% valid** (57/64), genuinely diverse
  drug-like SMILES — see the sanity-check output logged when this was first run.

## Two rough edges hit while smoke-testing, both fixed in `g2pt_prior.py`, neither a blocker

1. **`AutoTokenizer.from_pretrained(...)` + `tokenizer(...)` raises** `Exception: WordPiece
   error: Missing [UNK] token from the vocabulary` under this environment's `tokenizers`
   library version, when encoding even a trivial string like `'<boc>'`. Worked around by never
   calling the HF tokenizer's encode/decode convenience wrapper at all — `g2pt_prior.py` reads
   `vocab.json` directly for `stoi`/`itos`, which is all G2PT's own fixed-vocabulary format
   needs (no real subword tokenization happens; every token is already an atomic unit).
2. **The model's embedding/head matrices are padded wider than the real 112-token vocab**
   (`AutoConfig` reports `vocab_size=120`; the repo's own `model.py` comment says this padding
   is deliberate, "for efficiency", nanoGPT convention). Those extra rows are never trained or
   masked at inference, so unconditional sampling occasionally draws one — confirmed via a
   `KeyError` on first attempt, not assumed. `FrozenG2PT.real_token_mask` /
   `FrozenG2PT.mask_logits` fix this: every softmax in this pilot's code must mask the padded
   entries out, not just the eventual generated-molecule's decode step.
3. **Importing `vendor/G2PT/datasets_utils.py` directly is fragile**: it unconditionally imports
   `torch_geometric`, `networkx`, and the HuggingFace `datasets` library at module load time
   (needed by functions this pilot never calls), and a local `vendor/G2PT/datasets/` directory
   shadows the real `datasets` package if Python resolves the import while that directory is on
   `sys.path` (hit this running the vendor repo's own `example_sample_hf.py` from inside
   `vendor/G2PT/`). `g2pt_prior.py`'s `seq_to_smiles`/`_seq_to_mol` are copies of just the
   functions actually needed (`seq_to_molecule_with_partial_charges`, `get_smiles`,
   `check_valency`), not an import of the file.

## GPU-assignment bug during the first 24-run sweep (2026-09-04) -- 11/24 runs OOM'd

`scripts_g2pt/common.sh`'s original `gpu_for(count) = (count-1) % NUM_GPUS` assigned a GPU by
round-robining launch order, on the implicit assumption that jobs launched `NUM_GPUS` apart in
that order never overlap in time. False here: `db` vs `rtb` and different rewards finish at
different speeds (RTB in particular converges/diverges very differently from DB -- see
`molgpt`'s own training-curve finding), so two temporally-overlapping jobs were legitimately
assigned the same physical GPU while another GPU sat idle. 11 of the first 24 sweep runs OOM'd
this way (confirmed from the identical `torch.OutOfMemoryError` traceback in each failed run's
log, always showing a *different* process's memory footprint dominating the GPU, not the
failing process's own -- e.g. "Process 98873 has 28.84 GiB memory in use... this process has
2.44 GiB"). MolGPT's sweep never hit this in practice (its per-job memory footprint, ~1.8GB, is
small enough that two jobs sharing a GPU still fit in 32GB) -- same latent bug, just never
triggered a visible failure there.

Fixed in both legs' `common.sh` with a lock-directory-based `acquire_gpu`/`release_gpu` pair
(`mkdir` is atomic, so this is a correct mutex regardless of job timing) rather than a
timing-dependent scheme. The lock is acquired inside each launched job's own subshell, at actual
start time, and released via `trap ... EXIT` so a GPU frees up even if the job itself OOMs or
crashes.

**Consequence**: the first sweep's `nitrogen`/`osim` grid completed with only 13/24 configs
successful. Re-invoking `scripts_g2pt/01_train_guides.sh` (now with the fixed locking) retries
exactly the 11 missing ones -- the checkpoint-existence skip check treats a failed run (no
checkpoint saved) the same as a not-yet-attempted one, no special-casing needed.
