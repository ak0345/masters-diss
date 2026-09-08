"""Guide architectures: verbatim port of `hidden_guide.HiddenGuide` and
`gflow.LogitGuide`/`gflow.LogFlowHead` from the parent quetzal_gfn repo --
the same port already made once for `molgpt/molgpt_gfn/guides.py`, copied
again here rather than imported (see that file's docstring for the reasoning:
each of these directories is meant to become its own standalone repo).

Both guide classes are architecture-agnostic -- they operate on a
hidden-state tensor `h` and a `proj_logits` callable, never on anything
model-specific -- so this is a relocation, not an adaptation, the third time
running (Quetzal -> MolGPT -> G2PT). If these three copies ever drift, that's
a bug; there is exactly one correct implementation.

Not ported: `tempgain_guide.TempGainGuide` -- out of scope for this pilot
(hidden/base guides only), same as the MolGPT leg.
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
                 vocab_size=112, also_output_residual=False):
        super().__init__()
        self.d_model = d_model
        self._proj = proj_logits
        self.vocab_size = vocab_size

        blocks = []
        din = d_model
        for _ in range(max(layers - 1, 0)):
            blocks += [nn.Linear(din, hidden), nn.SiLU()]
            din = hidden
        blocks += [nn.Linear(din, d_model)]
        self.delta = nn.Sequential(*blocks)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        self.out_residual = None
        if also_output_residual:
            self.out_residual = nn.Sequential(
                nn.Linear(d_model, hidden), nn.SiLU(),
                nn.Linear(hidden, vocab_size))
            nn.init.zeros_(self.out_residual[-1].weight)
            nn.init.zeros_(self.out_residual[-1].bias)

    def guided_logits(self, h):
        d = self.delta(h)
        logits = self._proj(h + d)
        if self.out_residual is not None:
            logits = logits + self.out_residual(h)
        return logits

    def forward(self, h):
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
    """Scalar per-state flow *residual* f_theta(h) -> R. Full log-flow is
    log F(s) = logp_prior_partial(s) + f_theta(h_s). Zero-init the output.
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
    rollout builders. Ported verbatim (minus the TempGainGuide 2-arg branch,
    out of scope here):

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
    the atom-token logit dimensions -- every vocabulary entry where
    `atom_token_mask` is False comes back EXACTLY equal to `prior_logits` at
    that position, for every guide architecture, at every decoding step,
    unconditionally. Identical copy of `molgpt/molgpt_gfn/guides.py`'s
    function of the same name -- see that file's docstring for the full
    rationale (author instruction, 2026-09-04: "configure the 1D/2D guides
    to perturb only the atom-selection logits... to maintain the cleanest
    parity with Quetzal's atom-level logit intervention").

    For G2PT specifically this is a second, complementary restriction on top
    of `gflow_g2pt.py`'s existing step-level `is_atom_step` gate (which
    forces full identity -- not just atom-dimension identity -- outside
    atom-decision steps): the step gate answers "may the guide act at all
    right now," this answers "which logit entries may it move when it does."
    Combining both gives a strictly stronger guarantee than either alone,
    at negligible extra cost -- see `papers/current/decisions.md`,
    2026-09-04.

    `atom_token_mask`: `[vocab_size]` bool tensor, True at indices that are
    atom tokens (here: the `ATOM_*` vocabulary range). Broadcasts against
    `[..., vocab_size]` logits."""
    guided = guided_logits_dispatch(guide, prior_logits, h)
    return torch.where(atom_token_mask, guided, prior_logits)
