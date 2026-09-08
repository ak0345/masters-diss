"""Frozen G2PT prior.

Loads the published HuggingFace checkpoint, freezes every parameter, and
exposes hidden states per generation step plus the atom-decision-step gating
that makes Option-A-style node-type-only guidance possible (see
`../vendor/PROVENANCE.md` and `papers/current/decisions.md`, 2026-09-03
entries, for how this was verified against the actual G2PT source rather
than assumed).

Unlike `molgpt/molgpt_gfn/molgpt_prior.py`, this does NOT reimplement the
transformer forward pass by hand: the published checkpoint is a standard
`transformers.GPT2LMHeadModel` (see `vendor/G2PT/model.py`'s `to_hf()` --
that's what produced it), so `GPT2Model` already exposes exactly the tensor
we need (`last_hidden_state`, post-`ln_f`, pre-`lm_head`) as first-class,
well-tested HF machinery. Use that directly rather than re-deriving it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# G2PT's sequence grammar, traced from vendor/G2PT/datasets_utils.py's
# to_seq_by_bfs/to_seq_by_deg (encoding) and seq_to_mol (decoding), not
# assumed -- see papers/current/decisions.md, 2026-09-03 "G2PT leg uses
# Option A":
#
#   <boc> ATOM_t0 IDX_0 <sepc> ATOM_t1 IDX_1 <sepc> ... ATOM_tn IDX_n <eoc>
#   <bog> IDX_a IDX_b BOND_t <sepg> IDX_c IDX_d BOND_t <sepg> ... <eog>
#
# An atom-type token is legal, and only legal, immediately after <boc> or
# <sepc>. This is what lets the guide be gated to atom-decision steps only:
# it's a purely syntactic/positional rule (which token came immediately
# before), true regardless of whether the sequence generated so far is
# chemically sensible -- no fragile semantic-validity assumption involved.
_ATOM_STEP_PRECEDING_TOKENS = ("<boc>", "<sepc>")

_DEFAULT_VOCAB = Path(__file__).resolve().parent.parent / "vendor" / "G2PT" / "tokenizers" / "geom_drugs" / "vocab.json"


@dataclass
class G2PTConfig:
    # GEOM-Drugs retrain, 2026-09-04 -- see ../../shared_data/PROVENANCE.md
    # and vendor/PROVENANCE.md. model_name_or_path is a local directory (the
    # HF-converted from-scratch checkpoint), not a Hub id -- no HF checkpoint
    # exists for this dataset/vocab combination.
    model_name_or_path: str = "checkpoints/g2pt_geom_drugs_hf"
    vocab_path: str = str(_DEFAULT_VOCAB)
    # 15 heavy elements, exactly Quetzal's own geom_with_h list (edm_metrics.py)
    # minus H -- author instruction: both new legs share Quetzal's own vocab.
    atom_types: tuple = ("B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi")
    bond_types: tuple = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


class FrozenG2PT(nn.Module):
    """Wraps a published G2PT `GPT2LMHeadModel` checkpoint, frozen.

    `proj_logits` is `self.model.lm_head` -- a plain tied-embedding
    `nn.Linear(n_embd, vocab_size, bias=False)` -- so `HiddenGuide`/
    `LogitGuide` construct against it unmodified, same as the MolGPT leg.
    """

    def __init__(self, cfg: G2PTConfig, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.device = device

        with open(cfg.vocab_path) as f:
            self.stoi: dict[str, int] = json.load(f)
        self.itos: dict[int, str] = {i: s for s, i in self.stoi.items()}

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.n_embd = self.model.config.n_embd
        self.block_size = self.model.config.n_positions
        # The model's embedding/head matrices are padded to a rounder size
        # than the real 112-entry vocab (config reports a larger vocab_size,
        # e.g. 120) -- vendor/G2PT's own model.py comment: "padded up to
        # nearest multiple ... for efficiency", nanoGPT convention. Tensor
        # ops use the real (padded) width; token<->id lookups use `stoi`
        # (112 real entries) exclusively, so the padding rows are simply
        # never referenced.
        self.padded_vocab_size = self.model.lm_head.out_features
        # The checkpoint's embedding/head matrices are padded wider than the
        # real vocab (112 real tokens vs. e.g. 120 rows) and those extra rows
        # are never trained or masked at inference -- confirmed empirically
        # (see vendor/PROVENANCE.md): unconditional sampling occasionally
        # draws one of them, which has no entry in `itos` and would silently
        # break decoding. Mask them out of every softmax explicitly rather
        # than catching the decode KeyError after the fact.
        real_ids = torch.tensor(sorted(self.stoi.values()), device=device)
        self.real_token_mask = torch.zeros(self.padded_vocab_size, dtype=torch.bool, device=device)
        self.real_token_mask[real_ids] = True

        self.boc_id = self.stoi["<boc>"]
        self.eoc_id = self.stoi["<eoc>"]
        self.sepc_id = self.stoi["<sepc>"]
        self.bog_id = self.stoi["<bog>"]
        self.eog_id = self.stoi["<eog>"]
        self.sepg_id = self.stoi["<sepg>"]
        self.atom_ids = torch.tensor(
            sorted(self.stoi[f"ATOM_{a}"] for a in cfg.atom_types), device=device)
        self._atom_step_preceding_ids = {self.boc_id, self.sepc_id}

        # Dimension-wise atom-token mask for guides.masked_guided_logits_dispatch
        # (author instruction, 2026-09-04) -- complements is_atom_step's
        # step-level gate above, see that function's docstring.
        atom_mask = torch.zeros(self.padded_vocab_size, dtype=torch.bool, device=device)
        atom_mask[self.atom_ids] = True
        self.atom_token_mask = atom_mask

    def is_atom_step(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """prev_token_ids: [B] the token just emitted -> [B] bool, whether
        the NEXT token is an atom-type decision. Purely syntactic (see
        module docstring) -- true regardless of the sequence's chemical
        validity so far."""
        return (prev_token_ids == self.boc_id) | (prev_token_ids == self.sepc_id)

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids -> (B, T, n_embd), post-ln_f / pre-lm_head
        -- `GPT2Model.forward`'s `last_hidden_state` is already exactly this
        tensor (HF applies `ln_f` internally before returning it), so this
        is a direct call, not a reimplementation."""
        out = self.model.transformer(input_ids=idx)
        return out.last_hidden_state

    @property
    def proj_logits(self):
        return self.model.lm_head

    def encode_atom_type(self, symbol: str) -> int:
        return self.stoi[f"ATOM_{symbol}"]

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Fill the padded (non-real-token) vocab entries with -inf so they
        never get sampled. logits: (..., padded_vocab_size). Every caller
        that turns logits into a sampling/eval distribution (generation,
        rollout, flip diagnostics) must apply this -- see `real_token_mask`
        above for why."""
        return logits.masked_fill(~self.real_token_mask, float("-inf"))

    @torch.no_grad()
    def generate_unconditional(self, batch_size: int, max_len: Optional[int] = None,
                                 temperature: float = 1.0) -> list[str]:
        """Sequencing step-1 sanity check: reproduces
        vendor/G2PT/example_sample_hf.py's own sampling loop (bypassing its
        `AutoTokenizer.__call__`, which errors on this checkpoint's
        tokenizer.json under this transformers/tokenizers version -- see
        vendor/PROVENANCE.md -- by building input ids directly from `stoi`,
        the way the vendor script's OWN decode side already does via
        `batch_decode`-equivalent id->token lookup). Returns raw
        space-joined token-sequence strings (not SMILES) -- feed these to
        `reward_adapter`-style SMILES via a graph-decode step, done in
        `gflow_g2pt.py`, not here."""
        max_len = max_len or self.block_size
        x = torch.full((batch_size, 1), self.boc_id, dtype=torch.long, device=self.device)
        for _ in range(max_len - 1):
            logits = self.proj_logits(self.hidden_states(x))[:, -1, :] / temperature
            logits = self.mask_logits(logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
            if (nxt.squeeze(-1) == self.eog_id).all():
                break
        return [" ".join(self.itos[i] for i in row) for row in x.tolist()]


def _seq_to_mol(seq_str: str):
    """Ported from vendor/G2PT/datasets_utils.py's `seq_to_molecule_with_partial_charges`
    (the branch example_sample_hf.py uses for 'guacamol' checkpoints) --
    copied rather than imported because importing datasets_utils.py directly
    pulls in torch_geometric/networkx/the HF `datasets` library for
    functions this pilot never calls, and because a local `vendor/G2PT/datasets/`
    directory shadows the real `datasets` package when Python resolves
    imports from that directory (confirmed while smoke-testing this
    checkpoint -- see vendor/PROVENANCE.md), which would make importing the
    vendor file directly fragile depending on cwd."""
    from rdkit import Chem
    from rdkit.Chem import rdchem

    ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
    BOND_TYPE_RDKIT = {
        "BOND_SINGLE": rdchem.BondType.SINGLE,
        "BOND_DOUBLE": rdchem.BondType.DOUBLE,
        "BOND_TRIPLE": rdchem.BondType.TRIPLE,
        "BOND_AROMATIC": rdchem.BondType.AROMATIC,
    }

    tokens = seq_str.split()
    mol = Chem.RWMol()

    ctx_start = tokens.index("<boc>") + 1
    ctx_end = tokens.index("<eoc>")
    ctx_tokens = tokens[ctx_start:ctx_end + 1]

    id_atom_lookup = {}
    for i in range(0, len(ctx_tokens), 3):
        atom_type, atom_id = ctx_tokens[i], ctx_tokens[i + 1]
        atomic_symbol = atom_type.split("_")[1]
        atomic_num = Chem.Atom(atomic_symbol).GetAtomicNum()
        mol.AddAtom(Chem.Atom(atomic_num))
        id_atom_lookup[atom_id] = mol.GetNumAtoms() - 1

    bond_start = tokens.index("<bog>") + 1
    bond_end = tokens.index("<eog>")
    bond_tokens = [t for t in tokens[bond_start:bond_end] if t != "<sepg>"]

    for i in range(0, len(bond_tokens), 3):
        src_id, dest_id, bond_type = bond_tokens[i], bond_tokens[i + 1], bond_tokens[i + 2]
        if src_id not in id_atom_lookup or dest_id not in id_atom_lookup:
            continue
        bond_type_rdkit = BOND_TYPE_RDKIT[bond_type]
        mol.AddBond(id_atom_lookup[src_id], id_atom_lookup[dest_id], bond_type_rdkit)
        flag, atomid_valence = _check_valency(mol)
        if not flag:
            assert len(atomid_valence) == 2
            idx, v = atomid_valence
            an = mol.GetAtomWithIdx(idx).GetAtomicNum()
            if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def _check_valency(mol):
    from rdkit import Chem
    import re as _re
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e_sub = str(e)[str(e).find("#"):]
        atomid_valence = list(map(int, _re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def seq_to_smiles(seq_str: str) -> Optional[str]:
    """token sequence string -> canonical SMILES, or None if it doesn't parse
    to a valid molecule. Ported from vendor/G2PT/datasets_utils.py's
    `get_smiles` (largest-fragment selection + sanitize + canonicalize)."""
    smi, _reason = seq_to_smiles_classified(seq_str)
    return smi


def seq_to_smiles_classified(seq_str: str):
    """Like `seq_to_smiles`, but returns `(smiles_or_None, reason)` with
    `reason` in `'valid' | 'syntax_fail' | 'valence_fail'` -- author
    instruction, 2026-09-04: track syntactic parse failures separately from
    chemical valence failures (see reward_adapter.classify_smiles's
    docstring for the general rationale). For G2PT specifically,
    'syntax_fail' covers a grammar-level failure in `_seq_to_mol` itself
    (missing `<boc>`/`<eoc>`/`<bog>`/`<eog>`, a malformed node/edge section
    -- these raise before any RDKit sanitization is even attempted, since
    the token stream was never a well-formed graph to begin with);
    'valence_fail' covers `Chem.SanitizeMol` rejecting a structurally-built
    graph on chemistry grounds (bad valence, aromaticity)."""
    from rdkit import Chem

    try:
        mol = _seq_to_mol(seq_str)
    except Exception:
        return None, "syntax_fail"
    # Bug fixed 2026-09-05: a sequence with a syntactically well-formed but
    # EMPTY node section (e.g. `<boc><eoc><bog><eog>`, no ATOM_* tokens at
    # all) builds a zero-atom RWMol. `Chem.SanitizeMol` on a zero-atom mol
    # succeeds trivially (nothing to sanitize), and the very next line's
    # `GetMolFrags(..., asMols=True)` returns an EMPTY TUPLE for a zero-atom
    # mol -- `max((), default=mol, ...)` then silently falls back to
    # returning that same empty `mol`, `Chem.MolToSmiles` on it gives `""`,
    # and the whole thing was classified 'valid'. Confirmed by direct
    # inspection of a real dump: 99 of 101 "valid" entries in one config's
    # guided_smiles.txt were this exact empty string, only 2 were real
    # molecules -- every parse_rate/uniqueness/top-k reward number this
    # pilot reported for G2PT before this fix is contaminated by it (see
    # comparison_figures/, which was rebuilt after this fix; papers/current/
    # decisions.md has the full note). A zero-atom result is a degenerate
    # graph, not a valence problem, so it's classified syntax_fail here, not
    # valence_fail.
    if mol.GetNumAtoms() == 0:
        return None, "syntax_fail"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "valence_fail"
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        if largest.GetNumAtoms() == 0:
            return None, "syntax_fail"
        Chem.SanitizeMol(largest)
        return Chem.MolToSmiles(largest), "valid"
    except Exception:
        return None, "valence_fail"


def smoke_test(model_name_or_path: str, batch_size: int = 32, device: str = "cuda",
                min_valid_frac: Optional[float] = None) -> float:
    """Sequencing step 1: `python g2pt_prior.py`. See molgpt_prior.py's
    smoke_test for the same discipline -- informational by default, a hard
    gate only if `min_valid_frac` is passed (for unattended pipelines)."""
    prior = FrozenG2PT(G2PTConfig(model_name_or_path=model_name_or_path), device=device)
    seqs = prior.generate_unconditional(batch_size)
    smiles = [seq_to_smiles(s) for s in seqs]
    valid = [s for s in smiles if s is not None]
    valid_frac = len(valid) / len(seqs)
    print(f"generated {len(seqs)}")
    print(f"valid     {len(valid)}  ({valid_frac:.1%})")
    print("first 5 valid SMILES:")
    for s in valid[:5]:
        print(f"  {s}")
    if min_valid_frac is not None and valid_frac < min_valid_frac:
        raise SystemExit(
            f"[FATAL] valid_frac={valid_frac:.1%} is below --min_valid_frac={min_valid_frac:.1%}.")
    return valid_frac


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=smoke_test.__doc__)
    p.add_argument("--model_name_or_path", default="checkpoints/g2pt_geom_drugs_hf")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_valid_frac", type=float, default=None)
    args = p.parse_args()
    smoke_test(args.model_name_or_path, args.batch_size, args.device, args.min_valid_frac)
