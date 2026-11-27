"""Check a w8a8 -> bf16 dequant (utils/dequant_w8a8_to_bf16.py) against both of its references.

Two independent checks per sampled tensor:

1. **Round-trip** against the w8a8 source: the written bf16 must equal `int8 * weight_scale` bit-for-bit (after the
   bf16 cast). This catches shard/index mix-ups, wrong scale broadcasting and truncation bugs.

2. **Semantics** against the original bf16 model. A naive equality fails by construction: the dequant deliberately
   stays in the QuaRot basis, so each tensor is related to its bf16 counterpart by one of a few candidate relations.
   The script tries all of them and reports the best fit, instead of assuming which one should hold:

       identity      deq == bf16                      (not touched by QuaRot)
       in_gamma      deq == (bf16 * gamma) @ Q        (input side is hidden; preceding norm folded in)
       in_plain      deq == bf16 @ Q                  (input side is hidden, no fold)
       out           deq == Q.T @ bf16                (output side is hidden)
       folded_one    deq == 1                         (a norm whose gamma was folded away)
       block_in      each hidden-sized chunk of a wide mHC tensor rotated independently
       rot_q(...)    the q path carries a second, unpublished rotation R; an invariant of R is tested instead
       scaled_...    as above, with modelslim's uniform per-layer indexer rescale divided out

   Expected residuals: ~0.009 for int8-quantized tensors (that is the quantization noise), ~1e-3 or less for tensors
   modelslim left in FLOAT, and exactly 0 for untouched ones. A tensor whose best relation is wrong, or whose residual
   is far off for its class, is a real bug.

   **Known exception**, not a dequant bug: `layers.42.attn.indexer.wq_b.weight` exceeds the limit (~0.025 on the
   row-norm invariant). 1008 of its 8192 rows, spread over 60 of 64 heads, are up to 29 % below the layer's scalar,
   while layers 2 and 22 have zero rows off by even 2 %. The round-trip against w8a8 is exactly 0 there, so this is a
   property of modelslim's export of that one tensor, not of the dequant -- and it is part of what the int8 student
   costs, which is what arm C measures. See [[ds-w8a8-quarot-lora-basis]].

Also reports: key-set and dtype agreement over the whole checkpoint, and NaN/Inf over the sampled tensors.

Usage (inside lab-th_va25_qlora):
    python utils/check_w8a8_dequant.py \
        --dequant /data/lab-shared/timur/models_ds/DeepSeek-V4-Flash-Base-w8a8-dequant-bf16 \
        --w8a8    /data/models/DeepSeek-V4-Flash-Base-w8a8 \
        --bf16    /data/lab-shared/timur/models_ds/DeepSeek-V4-Flash-Base-bf16
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

AUX_SUFFIXES = ("_scale", "_offset")
NON_TENSOR_KEYS = ("model_quant_type", "group_size", "version", "metadata", "optional")
SAMPLE_LAYERS = (0, 1, 5, 21, 22, 42)


class Ckpt:
    def __init__(self, path, index_name=None):
        self.path = path
        if index_name is None:
            index_name = next(f for f in os.listdir(path) if f.endswith(".index.json"))
        with open(os.path.join(path, index_name)) as f:
            self.map = json.load(f)["weight_map"]
        self.handles = {}

    def _h(self, fname):
        h = self.handles.get(fname)
        if h is None:
            if len(self.handles) >= 6:
                self.handles.pop(next(iter(self.handles)))
            h = safe_open(os.path.join(self.path, fname), "pt")
            self.handles[fname] = h
        return h

    def has(self, key):
        return key in self.map

    def raw(self, key):
        return self._h(self.map[key]).get_tensor(key)

    def f32(self, key):
        return self.raw(key).float()

    def meta(self, key):
        sl = self._h(self.map[key]).get_slice(key)
        return sl.get_dtype(), tuple(sl.get_shape())


def rel(a, b):
    nb = b.norm()
    if nb == 0:
        return (a - b).norm().item()
    return ((a - b).norm() / nb).item()


def rel_scaled(a, b):
    """Residual after dividing out the least-squares optimal positive scalar.

    modelslim rescales indexer.wq_b / indexer.weights_proj by a uniform per-layer constant, which is invisible to
    DSA top-k. Fitting the scalar is what distinguishes "rescaled correctly" from "wrong tensor".
    """
    nb = (b * b).sum()
    if nb == 0:
        return rel(a, b), float("nan")
    c = ((a * b).sum() / nb).item()
    if c == 0:
        return rel(a, b), c
    return rel(a / c, b), c


def sample_keys(dq, bf):
    keys = ["embed.weight", "head.weight", "hc_head_base", "hc_head_fn", "hc_head_scale"]
    for L in SAMPLE_LAYERS:
        p = f"layers.{L}"
        keys += [
            f"{p}.attn_norm.weight", f"{p}.ffn_norm.weight",
            f"{p}.attn.q_norm.weight", f"{p}.attn.kv_norm.weight", f"{p}.attn.attn_sink",
            f"{p}.attn.wq_a.weight", f"{p}.attn.wq_b.weight", f"{p}.attn.wkv.weight",
            f"{p}.attn.wo_a.weight", f"{p}.attn.wo_b.weight",
            f"{p}.attn.compressor.wgate.weight", f"{p}.attn.compressor.wkv.weight",
            f"{p}.attn.compressor.norm.weight", f"{p}.attn.compressor.ape",
            f"{p}.attn.indexer.wq_b.weight", f"{p}.attn.indexer.weights_proj.weight",
            f"{p}.attn.indexer.compressor.wkv.weight",
            f"{p}.ffn.gate.weight", f"{p}.ffn.gate.bias",
            f"{p}.ffn.experts.0.w1.weight", f"{p}.ffn.experts.0.w2.weight", f"{p}.ffn.experts.0.w3.weight",
            f"{p}.ffn.experts.137.w1.weight", f"{p}.ffn.experts.255.w3.weight",
            f"{p}.ffn.shared_experts.w1.weight", f"{p}.ffn.shared_experts.w2.weight",
            f"{p}.ffn.shared_experts.w3.weight",
            f"{p}.hc_attn_base", f"{p}.hc_attn_scale", f"{p}.hc_ffn_fn",
        ]
    keys += ["mtp.attn.wq_a.weight", "mtp.ffn.experts.0.w1.weight", "mtp.ffn.shared_experts.w2.weight",
             "mtp.e_proj.weight", "mtp.enorm.weight", "mtp.norm.weight", "mtp.emb.tok_emb.weight"]
    return [k for k in keys if dq.has(k) and bf.has(k)]


def gamma_for(key, bf):
    """The norm whose scale QuaRot folds into this weight, if any."""
    if key == "head.weight":
        host = "norm.weight"                      # the final norm, folded into the LM head
    elif ".ffn." in key:
        host = key.split(".ffn.")[0] + ".ffn_norm.weight"
    elif ".attn." in key:
        host = key.split(".attn.")[0] + ".attn_norm.weight"
    else:
        return None
    return bf.f32(host) if bf.has(host) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dequant", required=True)
    ap.add_argument("--w8a8", required=True)
    ap.add_argument("--bf16", required=True)
    args = ap.parse_args()

    dq = Ckpt(args.dequant, "model.safetensors.index.json")
    w8 = Ckpt(args.w8a8, "quant_model_weights.safetensors.index.json")
    bf = Ckpt(args.bf16, "model.safetensors.index.json")
    with open(os.path.join(args.w8a8, "quant_model_description.json")) as f:
        desc = json.load(f)
    quant = {k for k, v in desc.items()
             if v == "W8A8_DYNAMIC" and not k.endswith(AUX_SUFFIXES) and k not in NON_TENSOR_KEYS}

    print("== key sets ==")
    print(f"dequant {len(dq.map)}  bf16 {len(bf.map)}")
    only_dq = sorted(set(dq.map) - set(bf.map))
    only_bf = sorted(set(bf.map) - set(dq.map))
    print(f"only in dequant: {len(only_dq)} {only_dq[:5]}")
    print(f"only in bf16:    {len(only_bf)} {only_bf[:5]}")

    print("\n== dtypes (dequant vs bf16, whole checkpoint) ==")
    mism = []
    for k in dq.map:
        if not bf.has(k):
            continue
        dd, ds = dq.meta(k)
        bd, bs = bf.meta(k)
        if ds != bs:
            mism.append((k, "shape", ds, bs))
        elif dd != bd:
            mism.append((k, "dtype", dd, bd))
    print(f"shape/dtype mismatches: {len(mism)} {mism[:5]}")

    Q = None
    with safe_open(os.path.join(args.w8a8, "optional", "quarot.safetensors"), "pt") as f:
        Q = f.get_tensor("global_rotation").float()

    keys = sample_keys(dq, bf)
    print(f"\n== {len(keys)} sampled tensors ==")
    print(f"{'tensor':62s} {'q':2s} {'roundtrip':>10s} {'relation':15s} {'resid':>9s}")

    bad = []
    for k in keys:
        d = dq.f32(k)

        # 1. round trip against the w8a8 source
        if k in quant:
            src = (w8.f32(k) * w8.f32(k + "_scale").reshape(-1, 1)).to(torch.bfloat16).float()
        else:
            src = w8.raw(k).to(torch.bfloat16).float() if dq.raw(k).dtype == torch.bfloat16 else w8.f32(k)
        rt = rel(d, src)

        # 2. semantics against the bf16 model
        b = bf.f32(k)
        cands = {}
        if d.shape == b.shape:
            cands["identity"] = rel(d, b)
        if d.ndim == 2 and d.shape[1] == Q.shape[0] and b.shape == d.shape:
            cands["in_plain"] = rel(d, b @ Q)
            g = gamma_for(k, bf)
            if g is not None and g.numel() == b.shape[1]:
                cands["in_gamma"] = rel(d, (b * g) @ Q)
        if d.ndim == 2 and d.shape[0] == Q.shape[0] and b.shape == d.shape:
            cands["out"] = rel(d, Q.T @ b)
        # a norm whose gamma was folded away is all-ones at whatever width it has (q_norm is q_lora_rank, not hidden)
        if d.ndim == 1:
            cands["folded_one"] = rel(d, torch.ones_like(d))
        # mHC hc_*_fn are (n, hc_mult * hidden): the hidden-sized chunks are rotated independently
        if d.ndim == 2 and b.shape == d.shape and d.shape[1] > Q.shape[0] and d.shape[1] % Q.shape[0] == 0:
            nb = d.shape[1] // Q.shape[0]
            cands["block_in"] = rel(d.reshape(-1, nb, Q.shape[0]), b.reshape(-1, nb, Q.shape[0]) @ Q)
        # q path: modelslim folds q_norm into wq_b and applies a second orthogonal rotation R on the q_lora axis.
        # R is not published, so test the invariant it must satisfy instead of the exact weights.
        if k.endswith("attn.wq_a.weight"):
            g = gamma_for(k, bf)
            a = (b * g) @ Q
            cands["rot_q(gram)"] = rel(d.T @ d, a.T @ a)
        if k.endswith("wq_b.weight"):
            host = k.rsplit(".wq_b.weight", 1)[0] + ".q_norm.weight"
            if bf.has(host):
                # R acts on wq_b's input axis, which leaves every output row's norm unchanged
                a = b * bf.f32(host)
                cands["rot_q(rownorm)"] = rel(d.norm(dim=1), a.norm(dim=1))
        # the DSA indexer carries an extra uniform per-layer positive scalar on top of the rotation; fit it out.
        # weights_proj is rotated by the global Q, wq_b by the unpublished q-axis R (so test row norms again).
        note = ""
        if k.endswith("indexer.weights_proj.weight") and d.shape == b.shape and d.shape[1] == Q.shape[0]:
            cands["scaled_in_plain"], c = rel_scaled(d, b @ Q)
            note = f"  scale {c:.4g}"
        elif k.endswith("indexer.wq_b.weight") and d.shape == b.shape:
            cands["scaled(rownorm)"], c = rel_scaled(d.norm(dim=1), b.norm(dim=1))
            note = f"  scale {c:.4g}"
        best, resid = min(cands.items(), key=lambda kv: kv[1]) if cands else ("none", float("nan"))

        flag = ""
        nan = (~torch.isfinite(d)).sum().item()
        if nan:
            flag = f"  !! {nan} non-finite"
            bad.append(k)
        if rt > 0:
            flag += f"  !! roundtrip {rt:.2e}"
            bad.append(k)
        lim = 0.02 if k in quant else 0.01
        if resid > lim:
            flag += f"  !! resid > {lim}"
            bad.append(k)
        print(f"{k:62s} {'Q' if k in quant else '-':2s} {rt:10.2e} {best:15s} {resid:9.5f}{note}{flag}")

    print("\n== verdict ==")
    if only_dq or only_bf or mism or bad:
        print(f"PROBLEMS: {len(only_dq) + len(only_bf)} key diffs, {len(mism)} shape/dtype, {len(set(bad))} tensors")
        raise SystemExit(1)
    print("all sampled tensors round-trip exactly and match the bf16 model under a QuaRot relation")


if __name__ == "__main__":
    main()
