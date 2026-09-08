# g2pt_gfn/

Guide training for the frozen G2PT prior. This mirrors the Quetzal leg's `gflow.py` module by
module, so a result differs between legs because the frozen model differs, not because the guide
code does.

| module | role |
|---|---|
| `g2pt_prior.py` | the frozen prior: a GPT-2 decoder over a graph-grammar vocabulary of node and edge tokens, loaded and held in eval mode |
| `guides.py` | guide architectures, a verbatim port of the Quetzal `HiddenGuide` and `LogitGuide` |
| `config.py` | `GFNConfig`, the full run configuration and its defaults |
| `gflow_g2pt.py` | the training loop, DB and RTB |
| `flip_g2pt.py` | flip diagnostics for a trained guide |
| `final_dump_g2pt.py` | sample and score a trained guide, same schema as the Quetzal dumps |
| `aggregate_g2pt.py` | pool run summaries into this leg's master table |
| `reward_adapter.py` | SMILES-native reward, the GuacaMol objectives and the nitrogen control |

The discriminator baseline lives here too. `fudge_data_g2pt.py` builds the labelled pairs, `train_fudge_g2pt.py` trains the
discriminator, and `fudge_generate_g2pt.py` samples and scores under it.

## Atom decisions

G2PT's grammar interleaves atom and non-atom tokens, so the guide is gated to
atom-decision steps with a precomputable per-step check (`atom_guide_only` in `config.py`, the
"Option A" gate the leg README explains). This is why its position-indexed and atom-indexed flip
curves are not two views of the same data, unlike MolGPT's token-per-atom stream.

## Why the guides are a verbatim port

The cross-architecture claim only holds if the guide is the same object in all three legs. The
architectures are therefore copied rather than reimplemented, and `config.py` keeps the same
field names as the Quetzal original so a configuration can be compared field by field.
