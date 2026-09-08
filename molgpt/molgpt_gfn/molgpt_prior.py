"""Frozen MolGPT prior.

Loads the pretrained checkpoint, freezes every parameter, and exposes hidden
states per generation step -- the thing a guide needs and the upstream
`generate/generate.py` doesn't surface (it only returns sampled ids).

See `../vendor/PROVENANCE.md` for what's been confirmed by reading
`vendor/molgpt` directly (LM-head shape, no KV-cache, tokenizer, pad/EOS
convention) versus what's still unverified because the checkpoint itself
hasn't been loaded yet -- see `../README.md`, "Blocked", before trusting
anything here beyond the static source reading.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_VENDOR_TRAIN = Path(__file__).resolve().parent.parent / "vendor" / "molgpt" / "train"
if str(_VENDOR_TRAIN) not in sys.path:
    sys.path.insert(0, str(_VENDOR_TRAIN))
from model import GPT, GPTConfig  # noqa: E402  (vendor/molgpt/train/model.py)

# Exact regex from vendor/molgpt/generate/generate.py:84 and train/dataset.py:39 --
# multi-character atoms (Cl, Br, [nH], ...) are single tokens, not char-level.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

PAD_TOKEN = "<"          # doubles as end-of-sequence -- see PROVENANCE.md
SEED_TOKEN = "C"         # devalab/molgpt's own unconditional seed (generate.py:57);
                          # not a dedicated BOS index, there isn't one.

_DEFAULT_STOI = Path(__file__).resolve().parent.parent / "vendor" / "molgpt" / "geom_drugs_stoi.json"


@dataclass
class MolGPTConfig:
    """GEOM-Drugs-retrained architecture (2026-09-04 -- see
    ../../shared_data/PROVENANCE.md and vendor/PROVENANCE.md's "GEOM-Drugs
    retrain" section). Superseded the original GuacaMol-unconditional
    checkpoint entirely, per author instruction: both new legs must train on
    GEOM-Drugs with Quetzal's own atom vocabulary, since Quetzal itself can't
    be rerun. `n_layer`/`n_head`/`n_embd` are unchanged (train.py's own
    argparse defaults, 8/8/256, independent of dataset); `vocab_size`,
    `block_size` and `scaffold_maxlen_for_mask` are all dataset-derived and
    come from `vendor/cond_gpt/weights/geom_drugs_unconditional_vocab_manifest.json`
    (`vocab_size=39`, `block_size=max_len=174`, `scaffold_maxlen_for_mask=
    scaffold_max_len=123`), not the old GuacaMol values (94, 100, 100)."""

    vocab_size: int = 39
    block_size: int = 174
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    stoi_path: str = str(_DEFAULT_STOI)
    # Buffer-sizing quirk, not an architecture choice: vendor/molgpt's
    # CausalSelfAttention sizes its causal-mask buffer as
    # `block_size + int(bool(num_props)) + int(scaffold_maxlen)` regardless
    # of whether `config.scaffold` is actually True (train/model.py's `num`
    # computation ignores `config.scaffold`, only `config.scaffold_maxlen`).
    # Our retrain (num_props=0, --scaffold never passed) still computed a
    # real scaffold_max_len=123 from the training CSV's `scaffold_smiles`
    # column (train.py always does, whether or not scaffold conditioning is
    # on), so the saved checkpoint's mask buffers are sized off block_size+123,
    # not block_size alone. The scaffold branch itself never executes
    # (`config.scaffold=False`), so this is inert padding, not a real
    # conditioning input -- but it must match at `load_state_dict(strict=True)`
    # time or every mask tensor mismatches. See vendor/PROVENANCE.md.
    scaffold_maxlen_for_mask: int = 123


class FrozenMolGPT(nn.Module):
    """Wraps devalab/molgpt's `GPT`, frozen, with hidden states surfaced.

    `proj_logits` is `self.model.head` directly: a plain
    `nn.Linear(n_embd, vocab_size, bias=False)` with no tied embeddings and no
    intermediate nonlinearity (confirmed in vendor/molgpt/train/model.py:125).
    That means `hidden_guide.HiddenGuide` and `gflow.LogitGuide` should be
    constructable against `proj_logits` unmodified -- see plan_molgpt.md,
    "Guide architecture reuse" -- once their exact constructor signatures are
    ported into `guides.py` (kept a separate module here since this is meant
    to become a standalone repo, not an import across a repo boundary that
    won't exist once it's moved).
    """

    def __init__(self, cfg: MolGPTConfig, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        with open(cfg.stoi_path) as f:
            self.stoi: dict[str, int] = json.load(f)
        self.itos: dict[int, str] = {i: s for s, i in self.stoi.items()}
        if len(self.stoi) != cfg.vocab_size:
            raise ValueError(
                f"stoi has {len(self.stoi)} entries but cfg.vocab_size={cfg.vocab_size}; "
                "these must match exactly or token indices silently misalign."
            )

        gconf = GPTConfig(
            cfg.vocab_size,
            cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            num_props=0,
            scaffold=False,
            scaffold_maxlen=cfg.scaffold_maxlen_for_mask,
            lstm=False,
            lstm_layers=0,
        )
        self.model = GPT(gconf)

        state = torch.load(checkpoint_path, map_location="cpu")
        # strict=True on purpose: a shape or key mismatch here is exactly
        # devalab/molgpt issue #18's failure mode (see PROVENANCE.md) and
        # should stop the run loudly, not silently drop/rename keys.
        self.model.load_state_dict(state, strict=True)

        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Convenience aliases used throughout molgpt_gfn/ -- computed once
        # here rather than by every caller, so there's one place that can be
        # wrong instead of several duplicated lookups.
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.seed_idx = self.stoi[SEED_TOKEN]

        # Atom-token mask for guides.masked_guided_logits_dispatch (author
        # instruction, 2026-09-04: guide reach restricted to atom-selection
        # logits, matching Quetzal's atom-only action space). A token is an
        # atom token iff it's bracket-atom notation ("[nH]", "[N+]", ...) or
        # a plain element symbol ("C", "Cl", "c", ...) -- every non-atom
        # token in this grammar (ring-closure digits, "%NN", "(", ")", bond
        # symbols "=#-/\:~@?*$", ".", the pad token "<") starts with neither
        # a letter nor "[", so this single rule cleanly separates the two
        # classes without hand-enumerating either one.
        atom_mask = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
        for tok, idx in self.stoi.items():
            if tok.startswith("[") or (tok and tok[0].isalpha()):
                atom_mask[idx] = True
        self.atom_token_mask = atom_mask

    def encode(self, smiles: str) -> list[int]:
        return [self.stoi[s] for s in SMILES_REGEX.findall(smiles)]

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids -> (B, T, n_embd), the post-LayerNorm,
        pre-head hidden state (i.e. `ln_f(x)`, the exact tensor `self.head` is
        applied to in the upstream `GPT.forward`).

        Reimplements the body of `GPT.forward` up to `ln_f` rather than
        hooking it, because the vendor file is kept unmodified (see
        vendor/PROVENANCE.md) and a forward hook on a `nn.Sequential` of
        blocks is more fragile than just calling them directly -- this method
        and `GPT.forward` will silently drift apart if the vendor model
        changes, so if vendor/molgpt is ever re-synced, diff this against the
        new `train/model.py` by hand.
        """
        b, t = idx.size()
        if t > self.model.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.model.block_size}")
        m = self.model
        tok = m.tok_emb(idx)
        pos = m.pos_emb[:, :t, :]
        typ = m.type_emb(torch.ones((b, t), dtype=torch.long, device=idx.device))
        x = tok + pos + typ  # m.drop is a no-op in eval() (dropout p=0.1, train-only)
        for block in m.blocks:
            x, _attn = block(x)
        return m.ln_f(x)

    @property
    def proj_logits(self):
        """The frozen LM head -- pass this straight into HiddenGuide/LogitGuide
        as `proj_logits` once `guides.py` exists (see module docstring)."""
        return self.model.head

    @torch.no_grad()
    def generate_unconditional(
        self,
        batch_size: int,
        steps: Optional[int] = None,
        temperature: float = 1.0,
        seed_token: str = SEED_TOKEN,
    ) -> list[str]:
        """Reproduces vendor/molgpt/generate/utils.py's `sample()` +
        vendor/molgpt/generate/generate.py's unconditional branch exactly
        (including its `.replace('<', '')` de-padding, not a stricter
        first-pad-truncation) -- this is the sequencing step-1 sanity check:
        "load the checkpoint standalone, generate unconditionally, confirm
        valid-ish SMILES come out" (plan_molgpt.md). It is NOT the guided
        rollout the guide trains against -- that lives in
        `gflow_molgpt.py`'s rollout builder, which needs per-step hidden
        states and guide-perturbed logits, not just terminal samples.
        """
        steps = self.cfg.block_size - 1 if steps is None else steps
        x = torch.tensor(self.encode(seed_token), dtype=torch.long, device=self.device)
        x = x.unsqueeze(0).repeat(batch_size, 1)
        for _ in range(steps):
            x_cond = x if x.size(1) <= self.cfg.block_size else x[:, -self.cfg.block_size:]
            logits = self.proj_logits(self.hidden_states(x_cond))[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return ["".join(self.itos[i] for i in row).replace(PAD_TOKEN, "") for row in x.tolist()]


def smoke_test(checkpoint_path: str, batch_size: int = 32, device: str = "cuda",
                min_valid_frac: float = None) -> float:
    """Sequencing step 1, run standalone: `python molgpt_prior.py <ckpt>`.
    Prints validity/uniqueness so a human confirms "valid-ish SMILES" before any
    guide code gets built on top of this checkpoint, and returns the valid
    fraction. By default this is purely informational -- plan_molgpt.md's
    issue-#18 fallback (retrain from scratch) is meant to be a judgment call
    for whoever runs this, not something to automate past silently. Pass
    `min_valid_frac` (or `--min_valid_frac` from the CLI) to turn that
    judgment call into a hard gate instead -- e.g. for an unattended pipeline
    that must not proceed to a 24-run sweep on top of a broken prior; this
    raises SystemExit(1) rather than continuing quietly."""
    from rdkit import Chem

    prior = FrozenMolGPT(MolGPTConfig(), checkpoint_path, device=device)
    smiles = prior.generate_unconditional(batch_size)
    valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in valid}
    valid_frac = len(valid) / len(smiles)
    print(f"generated {len(smiles)}")
    print(f"valid     {len(valid)}  ({valid_frac:.1%})")
    print(f"unique    {len(canon)} of {len(valid)} valid")
    print("first 5 valid SMILES:")
    for s in valid[:5]:
        print(f"  {s}")
    if min_valid_frac is not None and valid_frac < min_valid_frac:
        raise SystemExit(
            f"[FATAL] valid_frac={valid_frac:.1%} is below --min_valid_frac="
            f"{min_valid_frac:.1%}. Refusing to proceed on what looks like a "
            "broken/undertrained checkpoint -- see plan_molgpt.md's issue-#18 "
            "fallback before retrying.")
    return valid_frac


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=smoke_test.__doc__)
    p.add_argument("checkpoint_path")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_valid_frac", type=float, default=None,
                    help="if set, exit 1 when valid fraction falls below this (for unattended pipelines)")
    args = p.parse_args()
    smoke_test(args.checkpoint_path, args.batch_size, args.device, args.min_valid_frac)
