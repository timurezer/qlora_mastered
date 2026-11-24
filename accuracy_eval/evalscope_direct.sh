#!/bin/bash
# Evaluate an already-running server on full 0-shot GSM8K and MATH-500.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_NAME=${MODEL_NAME:-ds}
PORT=${PORT:-8061}
BATCH_SIZE=${BATCH_SIZE:-64}
WORK_DIR=${WORK_DIR:-$SCRIPT_DIR/logs/$MODEL_NAME}
TEST_DATA_DIR=$SCRIPT_DIR/../data/gsm8k/no_reasoning
export no_proxy=127.0.0.1,localhost
export NO_PROXY=$no_proxy

for BENCH in gsm8k math_500; do
    if [ "$BENCH" = gsm8k ]; then
        DATASET=general_qa
        MAX_TOKENS=4096
        DATASET_ARGS='{"general_qa":{"local_path":"'$TEST_DATA_DIR'","subset_list":["test_no_system"],"metric_list":["Rouge"]}}'
    else
        DATASET=math_500
        MAX_TOKENS=16384
        DATASET_ARGS='{}'
    fi

    evalscope eval \
        --model "$MODEL_NAME" \
        --eval-batch-size "$BATCH_SIZE" \
        --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
        --api-key EMPTY \
        --datasets "$DATASET" \
        --dataset-args "$DATASET_ARGS" \
        --generation-config '{"do_sample":false,"max_tokens":'$MAX_TOKENS',"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
        --work-dir "$WORK_DIR/$BENCH"
done

# general_qa does not compute our numerical GSM8K accuracy directly.
mkdir -p "$WORK_DIR/gsm8k/results"

python "$SCRIPT_DIR/extract_evalscope_outputs_gsm8k.py" \
    --path "$WORK_DIR/gsm8k" \
    --save_path "$WORK_DIR/gsm8k/results" \
    --data_path "$TEST_DATA_DIR/test_no_system.json"

python "$SCRIPT_DIR/get_accuracy.py" \
    --path "$WORK_DIR/gsm8k/results" \
    --save_path "$WORK_DIR/gsm8k/results/accuracy_results.xlsx" \
    --nums_compare
