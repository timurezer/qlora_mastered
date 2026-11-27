"""Dequantize a modelslim W8A8 DeepSeek-V4-Flash checkpoint to a plain bf16 HF checkpoint.

Why
    /data/models/DeepSeek-V4-Flash-{Base,0731}-w8a8 were exported by modelslim with a QuaRot step followed by
    per-output-channel symmetric int8 weight quantization (see DeepSeek-V4-Flash_best_practice.yaml in the
    checkpoint). This tool undoes only the int8 step:

        W_bf16 = (W_int8 - weight_offset) * weight_scale          weight_scale: (out, 1) fp32, offset: all zero

    The QuaRot rotation is deliberately **kept**. The result is a self-consistent bf16 model living in the rotated
    basis: ffn_norm is folded to 1, the hidden dimension is multiplied by Q. An adapter trained on this checkpoint
    is therefore already in the basis the w8a8 checkpoint is served in, and must NOT be passed through
    utils/rotate_lora_quarot.py afterwards.

    Everything modelslim left in FLOAT (norms, router gate, compressor / indexer projections, wo_a, wo_b, embed,
    head, attn_sink, ape, hc_* and the mHC *_scale parameters) is copied through unchanged.

Quantized vs float is read from quant_model_description.json, never from the key suffix: the mHC parameters
hc_head_scale / layers.N.hc_{attn,ffn}_scale end in "_scale" but are ordinary weights, not quantization scales.

Usage (inside lab-th_va25_qlora, ~1-2 h, CPU only, output ~554 GB):
    python utils/dequant_w8a8_to_bf16.py \
        --input  /data/models/DeepSeek-V4-Flash-Base-w8a8 \
        --output /data/lab-shared/timur/models_ds/DeepSeek-V4-Flash-Base-w8a8-dequant-bf16
    python utils/dequant_w8a8_to_bf16.py --input ... --output ... --dry-run    # print the shard plan only
"""

import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AUX_SUFFIXES = ("_scale", "_offset")
NON_TENSOR_KEYS = ("model_quant_type", "group_size", "version", "metadata", "optional")
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
TORCH_DTYPE = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
               "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32,
               "F32": torch.float32, "I64": torch.int64, "F64": torch.float64}


class ShardReader:
    """Open safetensors shards on demand, keeping a few handles alive."""

    def __init__(self, root, weight_map, max_open=4):
        self.root = root
        self.weight_map = weight_map
        self.max_open = max_open
        self.handles = {}

    def _handle(self, fname):
        h = self.handles.get(fname)
        if h is None:
            if len(self.handles) >= self.max_open:
                self.handles.pop(next(iter(self.handles)))
            h = safe_open(os.path.join(self.root, fname), "pt")
            self.handles[fname] = h
        return h

    def get(self, key):
        return self._handle(self.weight_map[key]).get_tensor(key)

    def meta(self, key):
        sl = self._handle(self.weight_map[key]).get_slice(key)
        return sl.get_dtype(), tuple(sl.get_shape())


def load_layout(root):
    with open(os.path.join(root, "quant_model_weights.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    with open(os.path.join(root, "quant_model_description.json")) as f:
        desc = json.load(f)

    quant = {
        k for k, v in desc.items()
        if v == "W8A8_DYNAMIC" and not k.endswith(AUX_SUFFIXES) and k not in NON_TENSOR_KEYS
    }
    missing = sorted(k for k in quant if k not in weight_map)
    if missing:
        raise SystemExit(f"{len(missing)} quantized keys absent from the index, e.g. {missing[:3]}")
    for suffix in AUX_SUFFIXES:
        absent = sorted(k for k in quant if k + suffix not in weight_map)
        if absent:
            raise SystemExit(f"{len(absent)} quantized keys without {suffix}, e.g. {absent[:3]}")

    aux = {k + s for k in quant for s in AUX_SUFFIXES}
    logical = [k for k in weight_map if k not in aux]
    return weight_map, quant, logical


def reference_dtypes(path):
    """Per-key dtype of a reference HF checkpoint, so the output matches it exactly.

    modelslim stores many tensors modelslim left in FLOAT at a different width than the official FP8->bf16 dequant
    did (norms and compressor weights as F32, tid2eid as I32). The DeepSeek4 converter was only ever run against the
    official output and guards on bfloat16, so the safe thing is to reproduce its dtypes rather than modelslim's.
    """
    ref = Ckpt_meta(path)
    return {k: ref.dtype(k) for k in ref.map}


class Ckpt_meta:
    def __init__(self, path):
        index_name = next(f for f in os.listdir(path) if f.endswith(".index.json"))
        with open(os.path.join(path, index_name)) as f:
            self.map = json.load(f)["weight_map"]
        self.path = path
        self.handles = {}

    def dtype(self, key):
        fname = self.map[key]
        h = self.handles.get(fname)
        if h is None:
            if len(self.handles) >= 6:
                self.handles.pop(next(iter(self.handles)))
            h = safe_open(os.path.join(self.path, fname), "pt")
            self.handles[fname] = h
        return h.get_slice(key).get_dtype()


def plan(reader, quant, logical, shard_bytes, ref_dtypes):
    """Group tensors into output shards, in checkpoint order, without loading payloads."""
    shards, cur, cur_bytes, total = [], [], 0, 0
    for key in logical:
        dtype, shape = reader.meta(key)
        numel = 1
        for d in shape:
            numel *= d
        out_dtype = ref_dtypes.get(key, "BF16" if key in quant else dtype)
        nbytes = numel * DTYPE_BYTES[out_dtype]
        if cur and cur_bytes + nbytes > shard_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(key)
        cur_bytes += nbytes
        total += nbytes
    if cur:
        shards.append(cur)
    return shards, total


def dequantize(reader, key):
    w = reader.get(key)
    if w.dtype != torch.int8:
        raise SystemExit(f"{key}: expected int8, got {w.dtype}")
    offset = reader.get(key + "_offset")
    if offset.abs().max().item() != 0:
        raise SystemExit(f"{key}: non-zero weight_offset; the export is not symmetric, check the dequant formula")
    scale = reader.get(key + "_scale").reshape(-1, 1).float()
    if scale.shape[0] != w.shape[0]:
        raise SystemExit(f"{key}: scale {tuple(scale.shape)} does not match weight {tuple(w.shape)}")
    return (w.float() * scale).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard-size-gib", type=float, default=12.0)
    ap.add_argument("--dtype-reference", default=None,
                    help="HF checkpoint whose per-key dtypes the output should reproduce "
                         "(use the official FP8->bf16 dequant of the same model)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output) and os.listdir(args.output):
        raise SystemExit(f"ABORT: {args.output} exists and is not empty")

    weight_map, quant, logical = load_layout(args.input)
    reader = ShardReader(args.input, weight_map)
    print(f"{len(logical)} tensors, {len(quant)} of them int8-quantized", flush=True)

    ref_dtypes = {}
    if args.dtype_reference:
        ref_dtypes = reference_dtypes(args.dtype_reference)
        missing = [k for k in logical if k not in ref_dtypes]
        if missing:
            raise SystemExit(f"{len(missing)} keys absent from the dtype reference, e.g. {missing[:3]}")
        print(f"dtype reference: {args.dtype_reference}", flush=True)

    shards, total = plan(reader, quant, logical, int(args.shard_size_gib * 1024 ** 3), ref_dtypes)
    n = len(shards)
    print(f"plan: {n} output shards, {total / 1024 ** 3:.1f} GiB total", flush=True)
    if args.dry_run:
        return

    os.makedirs(args.output, exist_ok=True)
    new_map, done, t0 = {}, 0, time.time()
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {}
        for k in keys:
            t = dequantize(reader, k) if k in quant else reader.get(k)
            want = TORCH_DTYPE[ref_dtypes[k]] if k in ref_dtypes else None
            tensors[k] = t.to(want) if want is not None and t.dtype != want else t
        save_file(tensors, os.path.join(args.output, fname), metadata={"format": "pt"})
        for k in keys:
            new_map[k] = fname
        del tensors
        done += len(keys)
        el = time.time() - t0
        print(f"[{i}/{n}] {fname}  {done}/{len(logical)} tensors  {el / 60:.1f} min elapsed, "
              f"eta {el / i * (n - i) / 60:.1f} min", flush=True)

    with open(os.path.join(args.output, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, f, indent=1)
    for name in COPY_FILES:
        src = os.path.join(args.input, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output, name))

    with open(os.path.join(args.output, "PROVENANCE.md"), "w") as f:
        f.write(
            "# DeepSeek-V4-Flash w8a8, int8-dequantized to bf16\n\n"
            f"Source: `{args.input}` (modelslim QuaRot + per-channel symmetric int8).\n"
            f"Produced by `utils/dequant_w8a8_to_bf16.py` on {time.strftime('%Y-%m-%d')}.\n\n"
            "Only the int8 step was undone: `W_bf16 = W_int8 * weight_scale` (all weight_offset are zero).\n"
            "**The QuaRot rotation is kept.** ffn_norm is folded to 1 and the hidden dimension is in the rotated\n"
            "basis, so an adapter trained on this checkpoint is already in the basis the w8a8 checkpoint is served\n"
            "in and must NOT be passed through `utils/rotate_lora_quarot.py`.\n"
        )
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
