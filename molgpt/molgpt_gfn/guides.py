"""Guide architectures: verbatim port of `hidden_guide.HiddenGuide` and
`gflow.LogitGuide`/`gflow.LogFlowHead` from the parent quetzal_gfn repo.

Both guide classes are already architecture-agnostic -- they operate on a
hidden-state tensor `h` and a `proj_logits` callable, never on anything
Quetzal-specific (no coordinates, no atomic-number vocab) -- so this is a
relocation, not an adaptation. Ported by hand rather than imported across the
repo boundary because `molgpt/` is meant to become its own repository (see
`../README.md`); if these ever need to change, check whether the drift should
go back into the parent repo's `hidden_guide.py`/`gflow.py` too.

Not ported: `tempgain_guide.TempGainGuide` (temperature/gain wrapper around a
base guide) -- plan_molgpt.md scopes this pilot to `hidden` and `base`
(residual) guides only, no tempgain.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HiddenGuide(nn.Module):
    """Perturbs the hidden state before the frozen projection.

        guided_logits(h) = proj_logits(h + delta(h))  [+ optional out_residual(h)]

    delta is zero-initialised, so this starts at the frozen prior's logits.
    Ported verbatim from quetzal_gfn/hidden_guide.py.
    """

    def __init__(self, d_model, proj_logits, hidden=512, layers=2,
                 vocab_size=94, also_output_residual=False):
        super().__init__()
        self.d_model = d_model
        # A reference to the projection, which the prior has already frozen; it
        # is deliberately not registered as a trainable parameter here.
        self._proj = proj_logits
        self.vocab_size = vocab_size

        blocks = []
        din = d_model
        for _ in range(max(layers - 1, 0)):
            blocks += [nn.Linear(din, hidden), nn.SiLU()]
            din = hidden
        blocks += [nn.Linear(din, d_model)]   # the output is a hidden-state delta
        self.delta = nn.Sequential(*blocks)
        # zero-init the last layer so delta(h) = 0 at initialisation
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        # optional additive residual on the output logits, also zero-init
        self.out_residual = None
        if also_output_residual:
            self.out_residual = nn.Sequential(
                nn.Linear(d_model, hidden), nn.SiLU(),
                nn.Linear(hidden, vocab_size))
            nn.init.zeros_(self.out_residual[-1].weight)
            nn.init.zeros_(self.out_residual[-1].bias)

    def guided_logits(self, h):
        """Hidden state in, guided logits out (the projection is already applied)."""
        d = self.delta(h)
        logits = self._proj(h + d)
        if self.out_residual is not None:
            logits = logits + self.out_residual(h)
        return logits

    def forward(self, h):
        """Return guided_logits - prior_logits, so call sites expecting an
        additive residual still compose to the guided logits via
        `prior + guide(h)`. Prefer guided_logits(h) where possible."""
        with torch.no_grad():
            prior_logits = self._proj(h)
        return self.guided_logits(h) - prior_logits


class LogitGuide(nn.Module):
    """Plain residual-on-logits guide: guided = prior_logits + guide(h).
    This is the pilot's `base` guide. Ported verbatim from gflow.py."""

    def __init__(self, d_model, vocab_size, hidden, layers):
        super().__init__()
        dims = [d_model] + [hidden] * (layers - 1)
        net = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU()]
        out = nn.Linear(dims[-1], vocab_size)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        net.append(out)
        self.net = nn.Sequential(*net)

    def forward(self, h):
        return self.net(h)


class LogFlowHead(nn.Module):
    """Scalar per-state flow *residual* f_theta(h) -> R. The full log-flow is
    log F(s) = logp_prior_partial(s) + f_theta(h_s), assembled in the DB loss.
    Zero-init the output so training starts at log F(s) = prior partial (i.e.
    the guide begins as an unbiased continuation of the frozen prior).
    Ported verbatim from gflow.py."""

    def __init__(self, d_model, hidden, layers):
        super().__init__()
        dims = [d_model] + [hidden] * (layers - 1)
        net = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU()]
        out = nn.Linear(dims[-1], 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        net.append(out)
        self.net = nn.Sequential(*net)

    def forward(self, h):
        return self.net(h).squeeze(-1)


def guided_logits_dispatch(guide, prior_logits, h):
    """The exact guide-call dispatch pattern used throughout quetzal_gfn's
    rollout builders (gflow.py's `_generate_guided`/`_db_rollout_states`/
    `_teacher_force_states`, and ablations/single_flip_ablation.py). Ported
    verbatim (minus the TempGainGuide 2-arg branch, out of scope here):

        None            -> prior_logits unchanged
        HiddenGuide      -> guide.guided_logits(h)          (applies proj itself)
        LogitGuide       -> prior_logits + guide(h)          (bare residual)
    """
    if guide is None:
        return prior_logits
    if hasattr(guide, "guided_logits"):
        return guide.guided_logits(h)
    return prior_logits + guide(h)


def masked_guided_logits_dispatch(guide, prior_logits, h, atom_token_mask):
    """Like `guided_logits_dispatch`, but the guide's effect is restricted to
    the atom-token logit dimensions: every vocabulary entry where
    `atom_token_mask` is False comes back EXACTLY equal to `prior_logits` at
    that position, for every guide architecture, at every decoding step,
    unconditionally -- not approximately restricted, not restricted only at
    steps a grammar heuristic guesses are atom-decisions, but pinned
    dimension-by-dimension.

    Author instruction, 2026-09-04: "configure the 1D/2D guides to perturb
    only the atom-selection logits... to maintain the cleanest parity with
    Quetzal's atom-level logit intervention." Quetzal's entire action space
    IS atom types, so its guide's reach is atom-selection-only by
    construction -- neither MolGPT's SMILES-token vocabulary nor G2PT's
    graph-token vocabulary has that property for free (both mix atom tokens
    with structural syntax -- ring-closure digits, branch parens, bond-order
    symbols, or, for G2PT, `IDX_*`/`BOND_*`/structural markers -- into one
    flat softmax). This function is what gives both legs that property back.

    `atom_token_mask`: `[vocab_size]` bool tensor, True at indices that are
    atom tokens. Broadcasts against `[..., vocab_size]` logits.

    `torch.where`'s backward pass sends zero gradient into the discarded
    (non-atom) dimensions -- same exact-zero-gradient property `gflow_g2pt.py`'s
    step-level gate already relied on, here applied per-dimension instead of
    per-step, and applicable to MolGPT's SMILES grammar where no clean
    step-level "next token will be an atom" rule exists the way it does for
    G2PT's rigid node-list/edge-list grammar (see
    `papers/current/decisions.md`, 2026-09-04, for why the two legs needed
    different gating strategies)."""
    guided = guided_logits_dispatch(guide, prior_logits, h)
    return torch.where(atom_token_mask, guided, prior_logits)
