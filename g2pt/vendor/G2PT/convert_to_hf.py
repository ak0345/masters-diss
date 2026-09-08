"""Convert a from-scratch nanoGPT-format `ckpt.pt` (vendor train.py's own
save format) to a local HuggingFace `GPT2LMHeadModel` directory, via
`GPT.to_hf()` (model.py) -- the exact same conversion the published
`xchen16/g2pt-guacamol-small-deg` checkpoint went through before being
uploaded to the Hub. Keeps `g2pt_prior.py`'s `AutoModelForCausalLM.from_pretrained`
loading path completely unchanged: only `G2PTConfig.model_name_or_path`
needs to point at the output directory instead of a Hub repo id.

Usage: `python convert_to_hf.py <ckpt.pt> <out_dir>`
"""
import sys

import torch

from model import GPT, GPTConfig


def main():
    ckpt_path, out_dir = sys.argv[1], sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = ckpt["model"]
    # nanoGPT sometimes prefixes keys with "_orig_mod." (torch.compile) --
    # vendor's own train.py resume path strips this, mirrored here.
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    hf_model = model.to_hf()
    hf_model.save_pretrained(out_dir)
    print(f"wrote HF model to {out_dir} (iter_num={ckpt.get('iter_num')}, "
          f"best_val_loss={ckpt.get('best_val_loss')})")


if __name__ == "__main__":
    main()
