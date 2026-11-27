"""Count compressor / indexer / nf4 keys in a converted mcore checkpoint (zeroed compress_ratios -> 0 compressor keys).
Usage: python check_ckpt_comp.py <ckpt_dir> [--bf16]
  --bf16: check an ordinary (non-QLoRA) bf16 conversion instead — compressor/indexer keys
          must be present and there must be NO uint8 (nf4-packed) tensors.
"""
import glob
import sys

import torch

expect_bf16 = "--bf16" in sys.argv[1:]
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/iter_*/mp_rank_*/model_optim_rng.pt"))
total = {"keys": 0, "compressor": 0, "indexer": 0, "uint8": 0}
for path in files:
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    c = {"keys": len(sd), "compressor": sum("compressor" in k for k in sd), "indexer": sum("indexer" in k for k in sd),
         "uint8": sum(getattr(v, "dtype", None) == torch.uint8 for v in sd.values())}
    print(path.split(root)[-1], c)
    for k in total:
        total[k] += c[k]
print("TOTAL", total, "files", len(files))
if expect_bf16:
    ok = total["compressor"] > 0 and total["indexer"] > 0 and total["uint8"] == 0
    print("OK" if ok else "BAD: bf16 checkpoint has uint8 (nf4) tensors or missing compressor/indexer keys")
else:
    ok = total["compressor"] > 0 and total["indexer"] > 0 and total["uint8"] > 0
    print("OK" if ok else "BAD: missing compressor/indexer/nf4 keys")
