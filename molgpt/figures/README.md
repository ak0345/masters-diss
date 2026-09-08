# figures/

Figure scripts for this leg. Rendered output is not included; each script takes `--out`.

| script | figure |
|---|---|
| `make_fig01_landscape.py` | terminal top-10, guided against the frozen prior |
| `make_fig02_positional.py` | sample flip rate by trajectory position |
| `make_fig03_training_curves.py` | DB against RTB over the training run |
| `make_fig06_flip_position_raw.py` | raw flip counts by position, unnormalised |
| `make_fig12_mpo_components.py` | per-component breakdown of each MPO score |
| `make_fig15_chemspace.py` | chemical space against the prior and GEOM-Drugs |
| `make_all.sh` | all of the above |

`figstyle_pilot.py` holds the shared palette and helpers.

In-figure titles are deliberately short or absent, since the manuscript caption carries the
description. Panels keep short identifiers only, because a caption cannot say which panel it
means without them.
