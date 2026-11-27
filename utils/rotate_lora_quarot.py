"""Rotate a DeepSeek-V4-Flash LoRA adapter into the basis of a QuaRot w8a8 checkpoint.

Why
    The w8a8 checkpoints (/data/models/DeepSeek-V4-Flash-{Base,0731}-w8a8) were exported by modelslim with a QuaRot
    step: the hidden dimension is multiplied by a fixed 4096x4096 rotation Q, and the ffn_norm scales (gamma) are
    folded into the expert weights:
        w1/w3_w8a8 = (w1/w3_bf16 * gamma) @ Q,    w2_w8a8 = Q^T @ w2_bf16,    ffn_norm_w8a8 = 1
    vllm-ascend does not rotate LoRA adapters, so an adapter trained on the bf16 / nf4 model must be converted:
        gate/up lora_A' = (lora_A * gamma) @ Q,    down lora_B' = Q^T @ lora_B
    (the other LoRA matrices act on the inner expert dimension and are copied unchanged; routed and shared experts
    alike). An unrotated adapter on a w8a8 checkpoint applies its update in the wrong coordinates.

Inputs
    --w8a8  the exact w8a8 checkpoint the adapter will be served on. Q is NOT computed: modelslim saves it inside the
            checkpoint as optional/quarot.safetensors (tensor "global_rotation"; referenced from
            quant_model_description.json under optional -> quarot -> rotation_map). Each export may have its own Q.
    --bf16  the matching bf16 model, for the ffn_norm scales gamma that the w8a8 export folded away.

Steps
    1. verify the folding on layers 0, 5, 42 (routed + shared expert) and on the embedding; abort if it does not hold
       (--check-only stops here);
    2. transform the hidden-side LoRA matrices;
    3. write a new adapter directory (--out); the input adapter is left untouched.

Usage
    python utils/rotate_lora_quarot.py --adapter <hf adapter> --out <hf adapter>_quarot \
        [--w8a8 /data/models/DeepSeek-V4-Flash-Base-w8a8] [--bf16 .../DeepSeek-V4-Flash-Base-bf16]
    python utils/rotate_lora_quarot.py --check-only --w8a8 <w8a8 dir> --bf16 <bf16 dir>
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CHECK_LAYERS = (0, 5, 42)
CHECK_TOLERANCE = 0.03  # relative error; the int8 quantization noise is ~0.01
LORA_KEY = re.compile(
    r"\.layers\.(\d+)\.mlp\.(?:experts\.\d+|shared_experts)\.(gate_proj|up_proj|down_proj)\.lora_([AB])\.weight$"
)


class Checkpoint:
    """Read single tensors (as float32) from a sharded HF safetensors checkpoint."""

    def __init__(self, path):
        self.path = path
        index_file = next(f for f in os.listdir(path) if f.endswith(".index.json"))
        with open(os.path.join(path, index_file)) as f:
            self.weight_map = json.load(f)["weight_map"]

    def get(self, key):
        with safe_open(os.path.join(self.path, self.weight_map[key]), "pt") as f:
            return f.get_tensor(key).float()

    def get_dequantized(self, key):
        """w8a8 weight: int8 values times the per-output-channel scale."""
        return self.get(key) * self.get(key + "_scale").reshape(-1, 1)


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def load_rotation(w8a8_dir):
    path = os.path.join(w8a8_dir, "optional", "quarot.safetensors")
    with safe_open(path, "pt") as f:
        return f.get_tensor("global_rotation").float()


def verify_folding(w8a8, bf16, Q):
    """Check that the w8a8 checkpoint is the bf16 model rotated by Q with ffn_norm folded in."""
    for layer in CHECK_LAYERS:
        gamma = bf16.get(f"layers.{layer}.ffn_norm.weight")
        assert w8a8.get(f"layers.{layer}.ffn_norm.weight").eq(1).all(), f"layer {layer}: ffn_norm is not folded"
        for expert in ("experts.0", "shared_experts"):
            w1, w2 = (f"layers.{layer}.ffn.{expert}.{w}.weight" for w in ("w1", "w2"))
            err_w1 = rel_err(w8a8.get_dequantized(w1), (bf16.get(w1) * gamma) @ Q)
            err_w2 = rel_err(w8a8.get_dequantized(w2), Q.T @ bf16.get(w2))
            print(f"check layer {layer} {expert}: w1 {err_w1:.4f}  w2 {err_w2:.4f}")
            assert err_w1 < CHECK_TOLERANCE and err_w2 < CHECK_TOLERANCE, "the w8a8 / bf16 pair does not match"

    embed_w8a8 = w8a8.get("embed.weight")[:2000]
    embed_bf16 = bf16.get("embed.weight")[:2000]
    err_plain, err_rotated = rel_err(embed_w8a8, embed_bf16), rel_err(embed_w8a8, embed_bf16 @ Q)
    print(f"check embed: w8a8 vs bf16 {err_plain:.4f}  vs bf16@Q {err_rotated:.4f}")
    assert err_rotated < CHECK_TOLERANCE, "the embedding is not rotated by Q"


def rotate_adapter(state_dict, bf16, Q):
    """Rotate the hidden-side LoRA matrices in place; returns (number rotated, set of layers seen)."""
    gammas = {}
    n_rotated = 0
    for key in list(state_dict):
        match = LORA_KEY.search(key)
        if not match:
            continue
        layer, proj, ab = int(match.group(1)), match.group(2), match.group(3)
        weight = state_dict[key].float()

        if proj in ("gate_proj", "up_proj") and ab == "A":  # reads the (rotated, un-scaled) hidden state
            if layer not in gammas:
                gammas[layer] = bf16.get(f"layers.{layer}.ffn_norm.weight")
            rotated = (weight * gammas[layer]) @ Q
        elif proj == "down_proj" and ab == "B":  # writes into the rotated residual stream
            rotated = Q.T @ weight
        else:  # inner expert dimension: unchanged
            continue

        state_dict[key] = rotated.to(torch.bfloat16).contiguous()
        n_rotated += 1
    return n_rotated, set(gammas)


def save_adapter(state_dict, adapter_dir, out_dir, w8a8_dir):
    os.makedirs(out_dir, exist_ok=True)
    save_file(state_dict, os.path.join(out_dir, "adapter_model.safetensors"))

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        config = json.load(f)
    config["base_model_name_or_path"] = w8a8_dir
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    report = os.path.join(adapter_dir, "conversion_report.json")
    if os.path.exists(report):
        shutil.copy(report, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", help="input HF/PEFT LoRA adapter dir (unrotated)")
    parser.add_argument("--out", help="output adapter dir")
    parser.add_argument("--check-only", action="store_true", help="only verify the w8a8 folding against bf16")
    parser.add_argument("--w8a8", default="/data/models/DeepSeek-V4-Flash-Base-w8a8")
    parser.add_argument("--bf16", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       "models_ds", "DeepSeek-V4-Flash-Base-bf16"))
    args = parser.parse_args()
    if not args.check_only and not (args.adapter and args.out):
        parser.error("--adapter and --out are required unless --check-only")

    w8a8, bf16 = Checkpoint(args.w8a8), Checkpoint(args.bf16)
    Q = load_rotation(args.w8a8)

    # The folding must hold for routed and shared experts before the adapter is touched.
    verify_folding(w8a8, bf16, Q)
    if args.check_only:
        return

    state_dict = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    n_rotated, layers = rotate_adapter(state_dict, bf16, Q)
    print(f"rotated {n_rotated} of {len(state_dict)} tensors, layers {min(layers)}..{max(layers)}")
    assert n_rotated == len(state_dict) // 2, "expected exactly half of the LoRA tensors to be rotated"

    save_adapter(state_dict, args.adapter, args.out, args.w8a8)
    print("saved", args.out)


if __name__ == "__main__":
    main()
