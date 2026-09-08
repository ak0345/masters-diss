"""Minimal stand-in for `moses.utils` (the `molecularsets/moses` package),
covering only what `vendor/molgpt` imports at module load time.

The real `moses` package (PyPI name `molsets`) pulls in `pomegranate`, which
fails to build here (an old C-extension against a toolchain mismatch) and is
not worth fighting for a function this pilot's training path never actually
calls: `train/utils.py` and `train/trainer.py` both `from moses.utils import
get_mol` unconditionally at import time, but `trainer.py` only calls it
inside the `if self.config.generate:` branch (trainer.py:163), and this
pilot's retrain run sets `TrainerConfig(..., generate=False)` — so the import
must resolve, but the function itself is dead code on this path.

`get_mol` below is copied verbatim from
`molecularsets/moses/moses/utils.py` (fetched 2026-09-03) rather than
reimplemented from memory, since even dead-on-this-path code should be
faithful if anything else ever exercises it.
"""
from rdkit import Chem


def get_mol(smiles_or_mol):
    """Loads SMILES/molecule into RDKit's object."""
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol
