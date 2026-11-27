"""rotate_lora_quarot.py when the bf16 reference model is available only as the FP8 original.

Why this exists
    rotate_lora_quarot.py needs a bf16 copy of the Base model for two things: the ffn_norm gammas that the
    w8a8 export folded away, and the w1/w2/embed tensors that verify_folding checks. models_ds/
    DeepSeek-V4-Flash-Base-bf16 has been reduced to config + tokenizer (6 MB), so neither is available
    there -- but /data/models/DeepSeek-V4-Flash-Base, the FP8 original it was dequantized from, has:
      * ffn_norm.weight, embed.weight and ffn.gate.weight in BF16 (dequantization never touched them, so
        these are byte-identical to what the deleted bf16 copy held);
      * ffn.*.w1/w2.weight in F8_E4M3 with 128x128 block scales under the sibling key '...w<n>.scale'.
    Reading the FP8 weights through rotate_lora_quarot.Checkpoint.get() would compare unscaled values and
    trip its assert, so this subclass applies the block scales. Everything else -- verify_folding,
    rotate_adapter, save_adapter -- is imported from that file and runs unmodified, so the full
    verification still gates the rotation.

usage: python utils/rotate_lora_quarot_fp8base.py --adapter <dir> --out <dir>_quarot
           [--w8a8 /data/models/DeepSeek-V4-Flash-Base-w8a8] [--fp8 /data/models/DeepSeek-V4-Flash-Base]
       python utils/rotate_lora_quarot_fp8base.py --check-only
"""
import argparse
import importlib.util
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_rq", os.path.join(_HERE, "rotate_lora_quarot.py"))
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)


class Fp8Checkpoint(rq.Checkpoint):
    """Same reader, but F8_E4M3 tensors are multiplied by their block scales."""

    def get(self, key):
        with safe_open(os.path.join(self.path, self.weight_map[key]), "pt") as f:
            t = f.get_tensor(key)
        if t.dtype != torch.float8_e4m3fn:
            return t.float()
        skey = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else key + ".scale"
        with safe_open(os.path.join(self.path, self.weight_map[skey]), "pt") as f:
            s = f.get_tensor(skey).float()
        t = t.float()
        assert t.ndim == 2 and s.ndim == 2, (key, t.shape, s.shape)
        br, bc = t.shape[0] // s.shape[0], t.shape[1] // s.shape[1]
        assert br * s.shape[0] == t.shape[0] and bc * s.shape[1] == t.shape[1], (key, t.shape, s.shape)
        return (t.reshape(s.shape[0], br, s.shape[1], bc) * s[:, None, :, None]).reshape(t.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--adapter")
    p.add_argument("--out")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--w8a8", default="/data/models/DeepSeek-V4-Flash-Base-w8a8")
    p.add_argument("--fp8", default="/data/models/DeepSeek-V4-Flash-Base")
    a = p.parse_args()
    if not a.check_only and not (a.adapter and a.out):
        p.error("--adapter and --out are required unless --check-only")

    w8a8, bf16 = rq.Checkpoint(a.w8a8), Fp8Checkpoint(a.fp8)
    Q = rq.load_rotation(a.w8a8)
    print(f"Q orthogonality: max|QQ^T - I| = {(Q @ Q.T - torch.eye(Q.shape[0])).abs().max().item():.2e}")
    rq.verify_folding(w8a8, bf16, Q)
    if a.check_only:
        return

    sd = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))
    n, layers = rq.rotate_adapter(sd, bf16, Q)
    print(f"rotated {n} of {len(sd)} tensors, layers {min(layers)}..{max(layers)}")
    assert n == len(sd) // 2, "expected exactly half of the LoRA tensors to be rotated"
    rq.save_adapter(sd, a.adapter, a.out, a.w8a8)
    print("saved", a.out)


if __name__ == "__main__":
    main()
