#!/bin/bash
# Full accuracy evaluation: start the vLLM server, wait for readiness, run
# full GSM8K + MATH-500, stop the server. Run INSIDE the inference container
# (th_va30_qlora_infer), from anywhere:
#
#   bash accuracy_eval/eval_pipeline.sh          # untuned Base-w8a8 (default)
#   LORA=1 bash accuracy_eval/eval_pipeline.sh   # run-D rotated adapter (lora-adapter)
#   MODEL_PATH=/data/models/DeepSeek-V4-Flash-0731-w8a8 bash accuracy_eval/eval_pipeline.sh
#
# Results go to accuracy_eval/logs/<model-name>/{gsm8k,math_500}/.
# Env overrides: PORT, BATCH_SIZE, MAX_MODEL_LEN, READY_TIMEOUT, LORA_PATH (with LORA=1).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

LORA=${LORA:-0}
PORT=${PORT:-8061}
READY_TIMEOUT=${READY_TIMEOUT:-900}   # model loading takes several minutes
mkdir -p "$SCRIPT_DIR/logs"
LOG=${LOG:-$SCRIPT_DIR/logs/serve_$(date +%Y%m%d_%H%M%S).log}

export no_proxy=127.0.0.1,localhost
export NO_PROXY=$no_proxy
export PORT

# Some containers (th_va30_qlora_infer) keep the python user bin off PATH.
command -v vllm >/dev/null || export PATH=/usr/local/python3.12.13/bin:$PATH

if [ "$LORA" = 1 ]; then
    export MODEL_NAME=lora-adapter
    SERVER_SCRIPT=serve_ds_w8a8_lora.sh
else
    export MODEL_NAME=ds
    SERVER_SCRIPT=serve_ds_w8a8.sh
fi

echo "Starting server ($SERVER_SCRIPT, port $PORT); log: $LOG"
# setsid: the server gets its own process group, so the trap can kill bash and
# vllm together; killing only the parent would orphan vllm holding the NPUs.
setsid bash "$SCRIPT_DIR/$SERVER_SCRIPT" > "$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill -TERM -"$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for the server (timeout ${READY_TIMEOUT}s)..."
t0=$(date +%s)
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ABORT: server exited during startup; see $LOG"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then
        break
    fi
    if (( $(date +%s) - t0 > READY_TIMEOUT )); then
        echo "ABORT: server not ready after ${READY_TIMEOUT}s; see $LOG"
        exit 1
    fi
    sleep 10
done
echo "Server is up after $(( $(date +%s) - t0 ))s."

bash "$SCRIPT_DIR/evalscope_direct.sh"

echo
echo "Eval pipeline OK."
echo "  MATH-500: $SCRIPT_DIR/logs/$MODEL_NAME/math_500/reports/"
echo "  GSM8K:    $SCRIPT_DIR/logs/$MODEL_NAME/gsm8k/results/"
