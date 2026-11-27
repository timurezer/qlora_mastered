#!/bin/bash
# Export the trained mcore QLoRA adapter (1-epoch distill run, iter 221) to an
# unrotated HF/PEFT adapter. CPU only.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RUN=${RUN:-deepseek4_flash_base_full_ep4_pp2_qlora_distill_1epoch}
IT=${IT:-221}
OUT=${OUT:-${RUN}_hf}

RUN="$RUN" IT="$IT" OUT="$OUT" bash "$ROOT/utils/export_iter_bf16.sh"
