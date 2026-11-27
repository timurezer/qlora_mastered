# Accuracy evaluation (vLLM + evalscope)

Copied from `qlora_clear/accuracy_eval/` (md5-verified 24 Nov 2026; only the default
`LORA_PATH` in `serve_ds_w8a8_lora.sh` points at this repo's adapter) — the same
scripts that evaluated the full-model smoke adapter. Run inside **`th_va30_qlora_infer`**
(MoE-LoRA patched vllm-ascend 0.30, evalscope 1.15.0, pyarrow 25.0.1) on node1; see the
container guide in `../AGENTS.md`.

## Flow

```bash
cd /data/lab-shared/timur/qlora_tests/qlora_mastered
# untuned base (Base-w8a8):
bash accuracy_eval/eval_pipeline.sh
# with a QuaRot-rotated LoRA adapter (served as "lora-adapter"):
LORA=1 LORA_PATH=$PWD/models_ds/<adapter>_quarot bash accuracy_eval/eval_pipeline.sh
```

- `eval_pipeline.sh` — starts the server (detached process group), waits for readiness,
  runs both benchmarks, kills the server on exit. Env: `PORT`, `BATCH_SIZE`,
  `MAX_MODEL_LEN`, `READY_TIMEOUT`, `LORA`, `LORA_PATH`.
- `serve_ds_w8a8.sh` — vLLM server for a w8a8 checkpoint (TP8 + expert parallel,
  `VLLM_ASCEND_LORA_MOE_OVERLAP=1`, graph mode). `serve_ds_w8a8_lora.sh` adds
  `--enable-lora` with `$LORA_PATH`.
- `evalscope_direct.sh` — full GSM8K (1319, `test_no_system`) + MATH-500, greedy,
  thinking off. Results in `logs/<model-name>/{gsm8k,math_500}/`.
- `extract_evalscope_outputs_gsm8k.py` + `get_accuracy.py` — GSM8K answer extraction and
  scoring.

## Notes

- Adapters trained on the bf16/nf4 checkpoint must be **QuaRot-rotated** before serving on
  Base-w8a8 (`qlora_clear/utils/rotate_lora_quarot_fp8base.py`, CPU-only, train container).
- **`get_accuracy.py` crashes on malformed numeric answers** (e.g. `float('0.1012012.00')`)
  — hit for real on 24 Nov. Use `../utils/score_gsm8k_results.py <results.json>` instead;
  it counts unparsable answers as wrong and prints the accuracy directly.
- Test data lives in `../data/gsm8k/no_reasoning/` (referenced relatively by the scripts);
  `test_random50_seed42.json` is a 50-problem quick subset. Of the 6 files there, only
  three are used: `test_no_system.jsonl` (evalscope input), `test_no_system.json`
  (reference answers for `extract_evalscope_outputs_gsm8k.py`), and the 50-problem
  subset. `test.json` / `test.jsonl` (same problems with an *empty* system message
  prepended) and `train.json` (train split) are unreferenced leftovers of the verbatim
  copy — kept for now, safe to delete if the empty-system variant is never needed.
- All 8 NPUs must be free (TP8). Server logs: `accuracy_eval/logs/serve_*.log`.
