# 2-node experiments (EP8 / PP2 / TP1, 16 cards)

Full-model DeepSeek-V4-Flash-Base LoRA and QLoRA on two 8x910B3 nodes, master branch
(`dev/qlora_support`). Counterparts of the single-node EP4/PP2 scripts in
`scripts_qlora/`, adapted from the old fork's 2-node script
(`qlora/scripts_qlora/lora_finetune_deepseek4_flash_0731_ep8_pp2_full_qlora_2nodes_gsm8k_comp.sh`)
with master's argument set (see `docs/ds_args.md`).

## Prerequisites (both nodes, same paths)

- This repo at the same path on both nodes; `MindSpeed-LLM` on branch
  `dev/qlora_support` (with `MindSpeed/` and `megatron/` in place).
- The training container (the `th_va30_qlora_train` equivalent: CANN 9.2.0, bnb NPU
  build, peft 0.20.0). Containers need inter-node networking for HCCL (host network or
  equivalent).
- `models_ds/DeepSeek-V4-Flash-Base-bf16` — the full bf16 Base (dequantized from FP8,
  `scripts_qlora/dequant_base_fp8_to_bf16.sh`, ~550 GB). Only needed on the node that
  runs the conversions.
- `data/data_train_ms_longalpaca/s<SEQ>/` for the perf sweep (and as default smoke data).

## 0. Rendezvous settings

All training/sweep scripts read three env vars (defaults at the top of each script,
edit there or export):

```bash
export MASTER_ADDR=<node0 IP reachable from node1>   # HCCL/RoCE interface
export MASTER_PORT=29510                             # any free TCP port
export NODE_RANK=0                                   # 0 on node0, 1 on node1
```

Both nodes always run the SAME command; only `NODE_RANK` differs. Start node1 first,
node0 (the rendezvous server) last.

## 1. Convert (single node, one process each)

```bash
nohup bash scripts_qlora_2nodes/ckpt_convert_deepseek4_flash_base_ep8_pp2_qlora.sh > scripts_qlora_2nodes/logs/log_convert_full_ep8pp2_nf4.txt 2>&1 &
nohup bash scripts_qlora_2nodes/ckpt_convert_deepseek4_flash_base_ep8_pp2_lora.sh  > scripts_qlora_2nodes/logs/log_convert_full_ep8pp2_bf16.txt 2>&1 &
```

Outputs: `models_ds/deepseek4_flash_base_full_ep8_pp2_{nf4,bf16}_mcore_comp` (16 rank
files each). The nf4 one uses the NPU quantizer (`DEVICES`, default 0-3); the bf16 one is
CPU-only and ~550 GB on disk. Both end with a key check (nf4: uint8 > 0; bf16: uint8 = 0).
Copy the resulting checkpoint dirs to the second node at the same path, or use shared
storage. Env overrides: `LOAD_DIR`, `SAVE_DIR`.

## 2. Train (both nodes)

```bash
# QLoRA (nf4 checkpoint), on node1 first, then node0:
nohup bash scripts_qlora_2nodes/qlora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh > scripts_qlora_2nodes/logs/log_train_qlora_2nodes.txt 2>&1 &

# LoRA (bf16 checkpoint) — heavier: ~34 GB of base weights per rank:
nohup bash scripts_qlora_2nodes/lora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh > scripts_qlora_2nodes/logs/log_train_lora_2nodes.txt 2>&1 &
```

Defaults: 5 iters, MBS 1, GBS 32, SEQ 2048, lr 3e-4, LoRA r16 on linear_fc1/linear_fc2,
fused DSA flags on, padded data. Env overrides: `TRAIN_ITERS MBS GBS SEQ_LEN DATA_PATH
CKPT_LOAD_DIR CKPT_SAVE_DIR NOPAD` plus QLoRA-only `BNB_DEQUANT_PATCH
EXPERT_DEQUANT_RECOMPUTE EXPERT_CHUNK_SIZE` (validated all-on config: `1 1 16`).
Loss should match between the two scripts at iteration 1 within nf4 noise (~0.1).
For a real distillation run point `DATA_PATH` at the tokenized teacher data and set
`NOPAD=1` (see `scripts_qlora/qlora_finetune_deepseek4_flash_base_distill_1epoch.sh`
for the proven 1-epoch settings: 221 iters, GBS 64, SEQ 4096).

## 3. Export the adapter to HF (single node, CPU)

Master's own `--save-lora-to-hf` rejects grouped-GEMM experts — use the
LoraAdapterTransformer wrapper (TP/PP/EP sizes are inferred from the checkpoint args):

```bash
RUN=deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke IT=5 \
OUT=deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke_hf \
BASE_NAME=models_ds/DeepSeek-V4-Flash-Base-bf16 \
CONVERTER=$PWD/LoraAdapterTransformer/examples/mindspeed-dsv4-flash/convert_mindspeed_dsv4_lora_to_vllm.py \
bash scripts_qlora/export_adapter_mcore2hf.sh
```

Produces a PEFT (vLLM-format) adapter under `models_ds/$OUT`. All 16 rank files must be
present in the run dir (gather them from both nodes into one directory first:
`iter_0000005/mp_rank_00_000..015`).

## 4. QuaRot-rotate the adapter for w8a8 serving (single node, CPU, train container)

Adapters trained on the bf16/nf4 checkpoint must be rotated before serving on the
w8a8 base. The tool lives in the qlora_clear tree (`rotate_lora_quarot_fp8base.py` reads
the FP8 original for the ffn_norm gammas); if that tree is absent on this server, copy
`utils/rotate_lora_quarot{,_fp8base}.py` from it into this repo's `utils/` first:

```bash
python utils/rotate_lora_quarot_fp8base.py \
  --adapter models_ds/deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke_hf \
  --out     models_ds/deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke_hf_quarot \
  --w8a8 /data/models/DeepSeek-V4-Flash-Base-w8a8 --fp8 /data/models/DeepSeek-V4-Flash-Base
```

CPU-only, verification (folding check) gates the rotation. Do NOT zero the shared-expert
branch — that recipe only applies to adapters trained with the broken seq-aux bug.

## 5. Serve + accuracy eval (single node, all 8 NPUs, infer container)

Runs in the `th_va30_qlora_infer` equivalent (MoE-LoRA patched vllm-ascend 0.30,
evalscope 1.15.0, pyarrow 25.0.1). bf16 does not fit on one node, so serving is always
Base-w8a8 (+ rotated adapter). See `accuracy_eval/README.md`:

```bash
# untuned base:
bash accuracy_eval/eval_pipeline.sh
# with the rotated adapter:
LORA=1 LORA_PATH=$PWD/models_ds/deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke_hf_quarot \
  bash accuracy_eval/eval_pipeline.sh
```

GSM8K + MATH-500, greedy, thinking off; results in `accuracy_eval/logs/<model>/`.
Score GSM8K with `utils/score_gsm8k_results.py` (not `get_accuracy.py`, which crashes on
malformed numeric answers).

## 6. Performance sweep (both nodes)

```bash
nohup bash scripts_qlora_2nodes/sweep_ctx_2nodes.sh > scripts_qlora_2nodes/logs/log_sweep2n.txt 2>&1 &
# LoRA arm:            MODE=lora TAG=lora nohup bash ... &
# QLoRA with the opts: TAG=optim EXTRA_ENV="BNB_DEQUANT_PATCH=1 EXPERT_DEQUANT_RECOMPUTE=1 EXPERT_CHUNK_SIZE=16" nohup bash ... &
```

Start on node1 first, node0 last, with identical `SEQS`/`MODE`/`TAG`/`EXTRA_ENV`.
Iterates `SEQS` (default 2048..16384, doubling), 3 iters each (s/iter = mean of iters
2-3, memory peaks at iter 1), stops at the first failure. Per-node artifacts:
`summary2n_<tag>_node{0,1}.txt`, `results2n_<tag>_node{0,1}.csv`,
`log2n_s<seq>_<tag>_node*.txt`, `npu_smi_mb2n_*.csv`. Stage 0 (node0) is the heavier
stage and sets the memory limit. Reference (old fork, EP8/PP2, 2 nodes): stock QLoRA max
ctx 5120, optimized 13312; master's single-node EP4/PP2 all-opts max ctx is 14336.

## Notes

- `sweep_ctx_2nodes.sh` reuses the samplers from `scripts_qlora/perf_ctx_sweep/`
  (`npu_smi_mb_sampler.py`, `npu_peak.py`); it must run where `npu-smi` and python3 are
  available (the training container has both).
- All scripts abort if the output/save dir already exists — move or remove old runs.
- `--no-shared-storage`: each node saves its own ranks locally; keep checkpoint paths
  node-local or identical on shared FS. Remember to gather all 16 ranks before export
  (step 3).
