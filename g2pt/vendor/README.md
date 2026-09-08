# vendor/

Upstream G2PT code, unmodified, kept so the frozen prior can be rebuilt from source.

`PROVENANCE.md` records where it came from and at what revision. The original `LICENSE` is kept
alongside the code and governs everything in this directory. Nothing here is this project's own
work.

**Excluded from this copy:** the upstream `datasets/`, `results/` and `wandb/` directories, and
any checkpoint or weight file. Those are bulk data rather than code, and the pipeline rebuilds
what it needs. Tokenizer and vocabulary files are kept, because the prior cannot be loaded
without them.
