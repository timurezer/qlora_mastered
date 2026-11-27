#!/bin/bash
# Dequantize the modelslim W8A8 Base checkpoint to a plain bf16 HF checkpoint.
# The QuaRot rotation is deliberately KEPT: the result is a self-consistent bf16
# model in the rotated (serving) basis, so an adapter trained on it must NOT go
# through scripts_qlora/rotate_adapter.sh afterwards.
# CPU only, ~1-2 h, output ~554 GB. Run inside the training container:
#   nohup bash scripts_qlora/dequant_w8a8_to_bf16.sh > logs_dequant_w8a8.txt 2>&1 &
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

INPUT=${INPUT:-/data/models/DeepSeek-V4-Flash-Base-w8a8}
OUT=${OUT:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-w8a8-dequant-bf16}
# Original bf16 reference for the semantic check (the FP8 dequant from step 3).
BF16_REF=${BF16_REF:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-bf16}

if [ -e "$OUT" ]; then echo "ABORT: $OUT exists"; exit 1; fi
mkdir -p "$ROOT/models_ds"

python "$ROOT/utils/dequant_w8a8_to_bf16.py" \
    --input "$INPUT" \
    --output "$OUT"

# Verify: round-trip against the w8a8 source, semantics against the bf16 reference.
python "$ROOT/utils/check_w8a8_dequant.py" \
    --dequant "$OUT" \
    --w8a8 "$INPUT" \
    --bf16 "$BF16_REF"

echo "w8a8 -> bf16 dequant OK: $OUT"
