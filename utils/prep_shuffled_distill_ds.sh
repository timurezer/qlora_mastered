#!/bin/bash
# Shuffle and tokenize the 14,090 teacher answers for run D. CPU only.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOKENIZER_PATH=${TOKENIZER_PATH:-/data/models/DeepSeek-V4-Flash-0731-bf16}
SAVE_DIR=$ROOT/data/data_train_ms_base_distill_teacher_ds_shuf

if [ -e "$SAVE_DIR" ]; then
    echo "ABORT: $SAVE_DIR exists; move it aside before rebuilding the dataset"
    exit 1
fi

python "$ROOT/utils/shuffle_distill_ds.py"
mkdir -p "$SAVE_DIR"

python "$ROOT/utils/convert_data2alpaca.py" \
    --path "$ROOT/data/base_distill/ds_teacher/shuf42/train_teacher.json" \
    --save_path "$SAVE_DIR/base_distill.parquet"

python "$ROOT/MindSpeed-LLM/preprocess_data.py" \
    --input "$SAVE_DIR/base_distill.parquet" \
    --tokenizer-name-or-path "$TOKENIZER_PATH" \
    --output-prefix "$SAVE_DIR/alpaca" \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 2000 \
    --enable-thinking false \
    --prompt-type deepseek4

echo "Dataset ready: $SAVE_DIR/alpaca"
