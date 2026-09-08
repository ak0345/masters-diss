# molgpt_gfn/

Guide training for the frozen MolGPT prior. This mirrors the Quetzal leg's `gflow.py` module by
module, so a result differs between legs because the frozen model differs, not because the guide
code does.

| module | role |
|---|---|
| `molgpt_prior.py` | the frozen prior: a GPT-style decoder over a SMILES character vocabulary, loaded and held in eval mode |
| `guides.py` | guide architectures, a verbatim port of the Quetzal `HiddenGuide` and `LogitGuide` |
| `config.py` | `GFNConfig`, the full run configuration and its defaults |
| `gflow_molgpt.py` | the training loop, DB and RTB |
| `flip_molgpt.py` | flip diagnostics for a trained guide |
| `final_dump_molgpt.py` | sample and score a trained guide, same schema as the Quetzal dumps |
| `aggregate_molgpt.py` | pool run summaries into this leg's master table |
| `reward_adapter.py` | SMILES-native reward, the GuacaMol objectives and the nitrogen control |

The discriminator baseline lives here too. `fudge_data_molgpt.py` builds the labelled (hidden state, eventual success) pairs and
`train_fudge_molgpt.py` trains the discriminator on them.

## Atom decisions

Not every generated token is an atom. MolGPT has no precomputable check for which
steps are atom decisions, so the diagnostics restrict post hoc to steps whose realised token is
an atom token. G2PT gates the same thing up front instead; the two conventions are why
`\Cref{fig:cmp02}` in the manuscript separates sequence position from atom index.

## Why the guides are a verbatim port

The cross-architecture claim only holds if the guide is the same object in all three legs. The
architectures are therefore copied rather than reimplemented, and `config.py` keeps the same
field names as the Quetzal original so a configuration can be compared field by field.
