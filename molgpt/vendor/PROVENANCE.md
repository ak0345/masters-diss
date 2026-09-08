# vendor/molgpt provenance

Vendored from `github.com/devalab/molgpt`, commit `72ff33ae747c0a4908b822732019a66e965a595a`
(2023-07-15), via a shallow clone on 2026-09-03. The `.git` directory was stripped after
cloning so this doesn't become a nested repo inside `molgpt/`'s own future git history — the
commit hash above is the only provenance record, keep it if this ever gets re-synced.

Citation (from the upstream README): Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar,
U. Deva (2021): *MolGPT: Molecular Generation using a Transformer-Decoder Model.* ChemRxiv.
Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1

Kept read-only / unmodified. `molgpt_gfn/` imports from this tree; if this ever needs a
change, copy the affected file out into `molgpt_gfn/` rather than editing here, so the vendor
tree stays a clean diff base against upstream.

## What's confirmed by reading this source (2026-09-03)

- **LM head** (`train/model.py:125`): `self.head = nn.Linear(config.n_embd, config.vocab_size,
  bias=False)`. Plain linear, no tied embeddings, no intermediate nonlinearity. It is applied
  to `self.ln_f(x)` — the pre-head hidden state a guide should hook is **post-LayerNorm**, not
  the raw residual stream.
- **No KV-cache.** `generate/utils.py`'s `sample()` recomputes the full forward pass over the
  (block-size-cropped) growing sequence at every step. Fine for a pilot at this model size, but
  don't assume caching exists when estimating wall-clock.
- **Guacamol-unconditional config**: `vocab_size=94`, `block_size=100`, `n_layer=8`,
  `n_head=8`, `n_embd=256`, `num_props=0`, `scaffold=False`, `lstm=False`. Vocab is
  `guacamol2_stoi.json`, char/regex-tokenized SMILES (`generate/generate.py:84`) — not
  character-level in the naive sense, multi-char tokens like `Cl`, `Br`, `[nH]` are single
  vocab entries. Index 16 is `<`, the pad token doubling as end-of-sequence: training targets
  right-pad every SMILES with `<` to `block_size`, so the model learns to predict `<` once the
  molecule is complete and keep predicting it for the rest of the block. There is no separate
  BOS token — generation seeds the sequence with the literal atom character `"C"|
  (`generate/generate.py:57`, `context = "C"`), not a special index.
- **Issue #18** (`RuntimeError: Error(s) in loading state_dict for GPT`) turned out to be a
  real, distinct instance of the same *class* of bug, not the one in the GitHub issue. What's
  in the issue thread is a mask-buffer shape mismatch on a **property-conditioned** checkpoint
  (`gua_tpsa_logp_sas.pt`) from passing the wrong `block_size`/`num_props` at load time. What we
  actually hit, on 2026-09-03, loading Kaggle's `virajbagal/ligflow-final-weights` dataset's
  `guacamol_nocond.pt` (the file its name claims is the unconditional checkpoint):
  ```
  Unexpected key(s) in state_dict: "prop_nn.weight", "prop_nn.bias".
  size mismatch for blocks.0.attn.mask: checkpoint [1,1,101,101] vs model [1,1,100,100]
  ```
  i.e. `guacamol_nocond.pt` is **not** the `num_props=0` model `train_guacamol.sh`'s first line
  documents. Diffing it against `guacamol_qed.pt` from the same dataset (also downloaded that
  session) showed **byte-identical state dicts** — `guacamol_nocond.pt` is a mislabeled
  duplicate of the QED-conditioned checkpoint, not a separate unconditional one. Confirmed by
  behaviour, not just tensor equality: feeding constant property values through the QED-shaped
  architecture at generation time showed length/validity swinging hugely with the constant
  (0.0 → 62.5% valid, mean length 95.7 tokens, mostly running to the block-size cap; 1.0 → 95.3%
  valid, mean length 35.1), consistent with a QED-style "how drug-like" conditioning signal, not
  an inert one. There is no genuinely-unconditional checkpoint in that Kaggle dataset to fall
  back to.
- **Retrain path used instead** (plan_molgpt.md's documented fallback): the vendor README's own
  `guacamol2.csv` Google Drive link (`drive.google.com/drive/folders/1LrtGru7...`) is also dead
  — checked directly on 2026-09-03, the folder renders empty in Drive's UI ("Place files here"),
  not a permissions wall. Rebuilt `vendor/molgpt/datasets/guacamol2.csv` from GuacaMol's own
  official released splits instead (`ndownloader.figshare.com/files/13612760` /
  `.../13612766`, md5-verified against the checksums BenevolentAI's `guacamol` repo README
  publishes), computing `scaffold_smiles` via RDKit's Murcko scaffold (functionally unused for
  an unconditional, non-scaffold run — `GPT.forward` only consumes it when `config.scaffold` is
  true — but computed for real rather than stubbed, since it costs little and removes any doubt).
  `train.py`'s hardcoded 94-token vocab (`whole_string`) matches `guacamol2_stoi.json` exactly,
  so this reconstruction tokenizes onto the same vocabulary the generation-side code already
  assumes; every row whose SMILES contains a token outside that vocab was dropped rather than
  guessed into the nearest token.
- **Two more environment gaps hit and fixed while actually launching the retrain** (both
  independent of the checkpoint/dataset issues above, both real code paths in this pilot, not
  vendor bugs to route around):
  1. `train/train.py` reads `train_data[args.props]` unconditionally, regardless of
     `--num_props` -- `--props` defaults to `['qed']`, so even a genuinely unconditional
     (`num_props=0`) run needs a real `qed` column in the CSV or it KeyErrors before training
     starts. Added one (`Chem.MolFromSmiles` + `QED.qed`, real values, not a placeholder --
     it's never fed to the model when `num_props=0`, but computing it for real costs little and
     removes any doubt).
  2. `train/utils.py` and `train/trainer.py` both `from moses.utils import get_mol`
     unconditionally at import time, though `trainer.py` only calls it inside `if
     self.config.generate:` (dead code on our `generate=False` path). The real `moses` package
     (PyPI `molsets`) pulls in `pomegranate`, which fails to build in this environment (an old
     C-extension against a toolchain mismatch) -- not worth fighting for an unused function.
     Added `vendor/moses_stub/moses/utils.py`, a `get_mol` copied verbatim from
     `molecularsets/moses/moses/utils.py` (fetched 2026-09-03), put on `PYTHONPATH` only for the
     `train/train.py` invocation.
  3. (Caught by `reward_adapter.py`, not this vendor tree, but the same shape of problem:)
     `guacamol`'s own `utils/chemistry.py` does `from scipy import histogram`, a re-export scipy
     has since dropped. `reward_fn.py` in the parent repo already carries a one-line shim for
     this (`if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram`) that got missed
     when trimming that file down for `reward_adapter.py` -- added back once it surfaced as a
     crash on the very first `guacamol`-reward smoke test.

### The actual retrain run (2026-09-03) and two more findings

Training completed cleanly: 10 epochs, loss 4.5 → 0.23, "Saving at epoch 10" printed, then the
run exited 1 anyway. **Not a real failure**: `train/train.py`'s very last line is
`df.to_csv(f'{args.run_name}.csv', ...)`, and `Trainer.train()` (`trainer.py`) only ever
`return`s a real `df` inside `if self.config.generate:` — our run's `TrainerConfig(...,
generate=False)` (the vendor script's own hardcoded default) means `train()` implicitly returns
`None`, so this line always `AttributeError`s on a normal training run. Confirmed the checkpoint
itself is unaffected: it saves earlier in the same function, well before this line runs.
`scripts_molgpt/run_pipeline.sh` now gates Stage 1 on checkpoint existence, not exit code, for
exactly this reason.

**Second finding, load-bearing**: loading the resulting checkpoint against the
`num_props=0, scaffold=False, scaffold_maxlen=0` config `molgpt_prior.py` originally used failed
with the *issue-#18 mask-shape mismatch again* — `[1,1,200,200]` in the checkpoint vs.
`[1,1,100,100]` expected. Root cause, read directly from `train/model.py`'s
`CausalSelfAttention.__init__`:
```python
num = int(bool(config.num_props)) + int(config.scaffold_maxlen)
```
This sizes the mask buffer using `config.scaffold_maxlen` **unconditionally** — it does not
check `config.scaffold` at all, only the forward pass does. `train/train.py` always computes a
real `scaffold_max_len` from the training CSV's `scaffold_smiles` column and always passes it
into `GPTConfig`, whether or not `--scaffold` was given. Our training log printed
`Scaffold max len:  100`, so the saved checkpoint's mask buffers are oversized to 200×200 even
though the scaffold-conditioning branch in `forward()` never executes (`config.scaffold=False`
throughout — confirmed by loading cleanly once `scaffold_maxlen=100` is passed to match, with
no other tensor affected). This means **issue #18's mask-mismatch failure mode has now been hit
three separate times in this project** for three unrelated reasons (a wrong CLI `block_size` on
someone else's checkpoint per the original GitHub issue; a mislabeled `num_props=1` checkpoint
on Kaggle; and this unconditional-but-still-scaffold-sized mask) — it is a generic symptom of
this specific model family's mask-buffer convention, not a one-off. `molgpt_prior.py`'s
`MolGPTConfig.scaffold_maxlen_for_mask` (default `100`) documents and reproduces this exactly;
change it only if a differently-preprocessed checkpoint prints a different `Scaffold max len` at
training time.
