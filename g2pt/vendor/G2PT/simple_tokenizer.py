"""Drop-in replacement for `AutoTokenizer.from_pretrained(f'tokenizers/{dataset}')`
used only for the `geom_drugs` dataset, added 2026-09-04.

Why this exists: `AutoTokenizer.from_pretrained(...)` + `tokenizer(text)` raises
`Exception: WordPiece error: Missing [UNK] token from the vocabulary` under
this environment's `tokenizers` library version -- confirmed while
smoke-testing `g2pt_prior.py` (see `../PROVENANCE.md`, "Two rough edges").
That file works around it for *inference* by reading `vocab.json` directly
and never calling the HF wrapper. `train.py`'s own data pipeline
(`datasets_utils.get_datasets` -> `pre_tokenize_function`) calls
`tokenizer(data['text'], padding='max_length', return_tensors='pt')`
directly, so the same bug would break training itself, not just inference --
this class is the same "read the vocab, do it by hand" workaround, applied
to the one call site training actually needs.

G2PT's own vocab is not real subword tokenization: every token
(`<boc>`, `IDX_17`, `ATOM_C`, `BOND_AROMATIC`, ...) is already atomic and
space-separated in the `" ".join(...)`-built text `to_seq_by_bfs`/
`to_seq_by_deg` produce (`datasets_utils.py`), so a plain whitespace split
plus a vocab lookup is exactly what the underlying WordLevel tokenizer.json
model was already doing -- nothing is lost by bypassing HF's wrapper.
"""
from __future__ import annotations

import json


class SimpleWordTokenizer:
    def __init__(self, vocab_path: str, model_max_length: int):
        with open(vocab_path) as f:
            self.stoi: dict[str, int] = json.load(f)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.pad_token_id = self.stoi["[PAD]"]
        self.model_max_length = model_max_length

    def __call__(self, texts, padding="max_length", return_tensors="pt"):
        assert padding == "max_length" and return_tensors == "pt", \
            "SimpleWordTokenizer only implements what pre_tokenize_function uses"
        import torch
        if isinstance(texts, str):
            texts = [texts]
        L = self.model_max_length
        input_ids = torch.full((len(texts), L), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), L), dtype=torch.long)
        for i, text in enumerate(texts):
            toks = text.strip().split(" ")
            if len(toks) > L:
                raise ValueError(
                    f"sequence has {len(toks)} tokens > model_max_length={L} -- "
                    "the block_size in configs/datasets/geom_drugs.py is too small "
                    "for this molecule; see prepare_geom_drugs.py's stats output.")
            ids = [self.stoi[t] for t in toks]
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}
