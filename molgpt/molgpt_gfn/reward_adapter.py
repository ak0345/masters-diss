"""SMILES-native reward adapter: a trimmed, ported copy of quetzal_gfn's
`reward_fn.build_reward_smiles`, covering only the two reward kinds this
pilot's scope uses (plan_molgpt.md: nitrogen-fraction control, then one
GuacaMol MPO task) -- not the full file's qed/logp/tpsa/isomer/similarity/
force/atom_stability/guacamol_component branches, which are either
geometry-dependent (irrelevant to MolGPT) or genuinely out of scope for this
pilot's grid.

MolGPT's whole reward path is simpler than Quetzal's here, worth noting per
plan_molgpt.md: Quetzal must run `rdDetermineBonds` to infer 2D bonds from a
sampled 3D geometry before any graph-based reward can be scored (a real
failure mode -- the paper's "decoder artifact" discussion traces an isolated
UMAP cluster to exactly this). MolGPT emits a SMILES string directly, so
`Chem.MolFromSmiles` either parses or it doesn't -- there is no
geometry-dependent bond-perception step to fail in a geometry-specific way.

Ported by hand rather than imported across the repo boundary -- same
reasoning as `guides.py` (see its docstring): `molgpt/` is meant to become
its own repository.
"""
from __future__ import annotations

import math

import scipy
import numpy as np
# guacamol's `guacamol/utils/chemistry.py` does `from scipy import histogram`,
# a top-level re-export scipy has since dropped. Ported verbatim from
# reward_fn.py's own shim (same line, same reasoning) -- without this, any
# `import guacamol...` (triggered lazily, the first time a "guacamol" reward
# is actually built) raises ImportError on this environment's scipy version.
if not hasattr(scipy, "histogram"):
    scipy.histogram = np.histogram

from rdkit import Chem

# ----------------------- validity classification -----------------------
# Author instruction, 2026-09-04: track syntactic parse failures separately
# from chemical valence failures, rather than lumping both into one
# "invalid" bucket the way a plain `Chem.MolFromSmiles(smi) is not None`
# check does. The two failure modes mean different things: a syntax failure
# says the token stream itself was malformed (unbalanced rings/branches, or
# -- for G2PT -- a token-grammar violation that never produced a parseable
# SMILES string at all); a valence failure says the stream WAS
# syntactically a well-formed molecular graph, RDKit could build it, but
# chemistry (an atom exceeding its valence, a bad aromaticity perception)
# rejects it. Conflating them hides whether a guide is breaking the prior's
# grammar or just pushing it toward chemically implausible structures --
# a materially different finding.


def classify_smiles(smi) -> str:
    """SMILES string (or None/empty) -> 'valid' | 'syntax_fail' | 'valence_fail'.

    `sanitize=False` first, so a structurally-parseable-but-chemically-invalid
    graph doesn't get folded into the syntax-failure bucket -- RDKit's
    combined parse+sanitize (the default `Chem.MolFromSmiles(smi)`) can't
    tell the two apart on its own, since it returns None either way."""
    if not smi:
        return "syntax_fail"
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return "syntax_fail"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return "valence_fail"
    return "valid"


# ----------------------- nitrogen-fraction reward -----------------------
# Ported from reward_fn._score_nitrogen_fraction, rewritten to read an RDKit
# Mol's atoms directly instead of going through the atoms/coords _FakeMol
# indirection that exists only to bridge Quetzal's (atomic-numbers, 3D
# coords) output shape -- MolGPT has no coords, so that step is unnecessary
# rather than adapted (per plan_molgpt.md, "Reward adapter").


def _score_nitrogen_fraction(mol: Chem.Mol) -> float:
    """Fraction of heavy atoms that are nitrogen, score in (0, 1], higher =
    more N-rich. Deliberately easy, monotone, atom-level target: the guide
    raises it purely by emitting N (Z=7) instead of C/O/F at each step -- the
    exact discrete decision it controls. Empty molecule -> 0.0 (caller maps
    to the invalid floor); a small epsilon keeps log() finite for an
    all-carbon molecule."""
    heavy = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) == 0:
        return 0.0
    n_count = sum(1 for z in heavy if z == 7)
    eps = 0.05
    return (n_count + eps) / (len(heavy) + eps)


# ----------------------- GuacaMol resolution -----------------------
# Ported verbatim from reward_fn.py's _guacamol_suite_v2 / _guacamol_scoring_fn
# / _norm / _GUACAMOL_LABEL_TO_NAME.

_GUACAMOL_LABEL_TO_NAME = {
    "osimertinib_mpo": "Osimertinib",
    "fexofenadine_mpo": "Fexofenadine",
    "ranolazine_mpo": "Ranolazine",
    "perindopril_mpo": "Perindopril",
    "amlodipine_mpo": "Amlodipine",
    "sitagliptin_mpo": "Sitagliptin",
    "zaleplon_mpo": "Zaleplon",
    "valsartan_smarts": "Valsartan",
    "deco_hop": "Deco Hop",
    "scaffold_hop": "Scaffold Hop",
    "albuterol_similarity": "Albuterol",
    "celecoxib_rediscovery": "Celecoxib",
    "mestranol_similarity": "Mestranol",
    "thiothixene_rediscovery": "Thiothixene",
    "troglitazone_rediscovery": "Troglitazone",
    "median1": "Median molecules 1",
    "median2": "Median molecules 2",
    "isomers_c7h8n2o2": "C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "C9H10N2O2PF2Cl",
    "isomers_c11h24": "C11H24",
    # raw sb fn names -> display name (so old callers keep working)
    "hard_osimertinib": "Osimertinib",
    "hard_fexofenadine": "Fexofenadine",
    "amlodipine_rings": "Amlodipine",
    "perindopril_rings": "Perindopril",
    "sitagliptin_replacement": "Sitagliptin",
    "zaleplon_with_other_formula": "Zaleplon",
    "decoration_hop": "Deco Hop",
    "median_camphor_menthol": "Median molecules 1",
    "median_tadalafil_sildenafil": "Median molecules 2",
}

_GUACAMOL_V2_SUITE = None


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _guacamol_suite_v2():
    global _GUACAMOL_V2_SUITE
    if _GUACAMOL_V2_SUITE is None:
        from guacamol.benchmark_suites import goal_directed_benchmark_suite
        _GUACAMOL_V2_SUITE = goal_directed_benchmark_suite("v2")
    return _GUACAMOL_V2_SUITE


def _guacamol_scoring_fn(name):
    """Return a scoring object with a .score(smiles)->float method for the
    named GuacaMol v2 goal-directed benchmark. Accepts a raw
    standard_benchmarks function name (e.g. "hard_osimertinib",
    "perindopril_rings" -- what scripts_molgpt/01_train_guides.sh passes),
    a display name, or a friendly/heatmap label."""
    try:
        from guacamol import standard_benchmarks as sb
        fn = getattr(sb, name, None)
        if callable(fn):
            bench = fn()
            for attr in ("objective", "wrapped_objective"):
                obj = getattr(bench, attr, None)
                if obj is not None and hasattr(obj, "score"):
                    return obj
    except Exception:
        pass  # fall through to suite resolution

    target = _GUACAMOL_LABEL_TO_NAME.get(name, name)
    tnorm = _norm(target)
    suite = _guacamol_suite_v2()

    for bench in suite:
        if _norm(bench.name) == tnorm:
            obj = getattr(bench, "objective", None) or getattr(bench, "wrapped_objective", None)
            if obj is not None and hasattr(obj, "score"):
                return obj

    hits = [b for b in suite if tnorm in _norm(b.name)]
    if len(hits) == 1:
        obj = getattr(hits[0], "objective", None)
        if obj is not None and hasattr(obj, "score"):
            return obj

    avail = ", ".join(sorted(b.name for b in suite))
    raise AttributeError(
        f"Could not resolve guacamol benchmark {name!r}. "
        f"Available v2 benchmarks: {avail}")


# ----------------------- Factory -----------------------


def build_reward_smiles(cfg):
    """Return a callable log_reward(smi: str) -> float (base log reward,
    beta=1; the trainer applies cfg.reward_beta separately, matching
    reward_fn.py's convention).

    cfg.reward selects the kind: "nitrogen_count" | "guacamol" (needs
    cfg.reward_smiles, e.g. "hard_osimertinib"). Invalid/unparseable SMILES
    return cfg.invalid_logr, never raise.
    """
    kind = cfg.reward
    floor = cfg.invalid_logr

    def _log(score):
        return math.log(score) if score > 0 else floor

    if kind == "nitrogen_count":
        def log_reward(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return floor
            try:
                s = _score_nitrogen_fraction(mol)
            except Exception:
                return floor
            return _log(s)
        return log_reward

    if kind == "guacamol":
        if not cfg.reward_smiles:
            raise ValueError("cfg.reward=='guacamol' needs cfg.reward_smiles set "
                              "(e.g. 'hard_osimertinib').")
        scorer = _guacamol_scoring_fn(cfg.reward_smiles)

        def log_reward(smi):
            if Chem.MolFromSmiles(smi) is None:
                return floor
            try:
                s = float(scorer.score(smi))
            except Exception:
                return floor
            return _log(s)
        return log_reward

    raise ValueError(
        f"Unknown cfg.reward={kind!r}. This pilot's reward_adapter.py only "
        "ports 'nitrogen_count' and 'guacamol' -- see the module docstring "
        "for why the rest of reward_fn.py's kinds weren't ported.")
