# DeepSeek-V4-Flash-Base → W8A8 (2026-10-22)

> Copy of `quant/quantizaiton/` (the `quant` project), kept here because this w8a8 checkpoint is the student of the
> arm C experiment ([../../docs/dequant_model_training.md](../../docs/dequant_model_training.md)). The originals there
> are authoritative; update them first, then re-copy `quant_dsv4_flash_base_w8a8.sh`,
> `deepseek_v4_flash_base_w8a8.yaml` and this file.

| | |
|---|---|
| Input | `/data/models/DeepSeek-V4-Flash-Base` (FP8 block-quantized, 43 layers + 1 MTP) |
| Output | node2 `/data/lab-shared/timur_quant/models_quant/DeepSeek-V4-Flash-Base-w8a8` (280 GB, 74 shards) |
| Tool | msmodelslim 26.1.0 in `lab-th_glm_quant` (node2), 8 NPUs, ~16 min |
| Config | [deepseek_v4_flash_base_w8a8.yaml](deepseek_v4_flash_base_w8a8.yaml) |
| Script | [quant_dsv4_flash_base_w8a8.sh](quant_dsv4_flash_base_w8a8.sh) (log: `quant_dsv4_flash_base_w8a8.log` next to it on node2) |

## Recipe

The same recipe as `/data/models/DeepSeek-V4-Flash-0731-w8a8`, taken from the yaml saved in that folder
(`DeepSeek-V4-Flash-DSpark_best_practice.yaml`):

1. QuaRot rotation, block size 32.
2. W8A8 dynamic: int8 weights per-channel, int8 activations per-token, symmetric, minmax.
   - Quantized: `wq_a`, `wq_b`, `wkv`, `indexer.wq_b`, routed and shared experts (`w1/w2/w3`).
   - Kept bf16: `wo_a`, `wo_b`, compressors, `indexer.weights_proj`, `ffn.gate`, norms, `embed`, `head`,
     MTP `e_proj`/`h_proj`.
3. Calibration: `qwen3_cot_w4a4.json` (msmodelslim `lab_calib`).

The FP8 weights are dequantized to bf16 by the msmodelslim adapter while loading; no separate conversion step.

## Differences from the msmodelslim default

msmodelslim's `lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml` (`--quant_type w8a8`) differs from the
0731 recipe:

| | Default | 0731 recipe (used) |
|---|---|---|
| `flex_smooth_quant` | yes | no |
| Calibration set | `mix_calib.jsonl` | `qwen3_cot_w4a4.json` |

0731-w8a8 has no smoothing: its `attn_norm` / `ffn_norm` are exactly 1.0 after QuaRot fusion.

Other changes against the 0731 yaml:
- Its DSpark-only `mtp.*` block was dropped; `*attn*` / `*ffn*` already match the Base model's `mtp.0`.
- `--model_type DeepSeek-V4-Flash` (Base has a standard MTP layer, not DSpark).

## Checks

- Log ends with `SUCCESS`, no `ERROR` lines.
- `quant_model_description.json` has the same per-layer quant types as 0731-w8a8.
- Not yet loaded in vLLM-Ascend.

## Caveats

- msmodelslim asks for `transformers==4.48.2`; the container has 5.14.1. Not downgraded (container is shared with
  the GLM server); the DeepSeek-V4 adapter uses its own model code.
- Output `config.json` keeps `"expert_dtype": "fp8"` from Base (0731-w8a8 has `fp4`). May need editing if vLLM
  reads it with `--quantization ascend`.
- No `generation_config.json` (Base has none).
