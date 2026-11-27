#!/bin/bash
# Rotate the exported run-D adapter into the Base-w8a8 serving basis. CPU only.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

ADAPTER=${ADAPTER:-$ROOT/models_ds/deepseek4_flash_base_bf16_teacher_hf}
OUT=${OUT:-${ADAPTER}_quarot}

python "$ROOT/utils/rotate_lora_quarot_fp8base.py" \
    --adapter "$ADAPTER" \
    --out "$OUT"
