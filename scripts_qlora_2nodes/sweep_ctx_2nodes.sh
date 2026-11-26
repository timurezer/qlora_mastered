#!/bin/bash
# 2-node context-length performance sweep for the full DeepSeek-V4-Flash (master branch,
# dev/qlora_support), EP8 / PP2 / TP1, 16 cards, LongAlpaca exact-length data.
# Port of scripts_qlora/perf_ctx_sweep/sweep_ctx.sh to 2 nodes; counterpart of the old
# fork's scripts_qlora/ctx2n study (CONTEXT_LENGTH_RESULTS_2NODES.md).
#
# For each length in $SEQS it runs the 2-node training script (MODE selects which) with
# TRAIN_ITERS=3, MBS=1, GBS=32 on data/data_train_ms_longalpaca/s<SEQ> (memory peaks in
# iteration 1; s/iter = mean of iterations 2-3). STOPS at the first failing run (OOM or
# otherwise) — the remaining lengths cannot fit either.
#
# While each run trains, npu_smi_mb_sampler.py records exact per-card HBM MB every 2 s;
# npu_peak.py reports per-card peaks with the idle ~3.4 GB subtracted (IDLE_MB).
#
# MUST BE STARTED ON BOTH NODES with identical SEQS/MODE/TAG/EXTRA_ENV — only NODE_RANK
# differs (NODE_RANK only). Start node1 first, node0 last. Each node sweeps independently
# (HCCL rendezvous absorbs the timing skew) and writes its own summary/csv:
# stage 0 = node0 (ranks 0-7, the heavier stage that sets the memory limit),
# stage 1 = node1 (ranks 8-15).
#
# Run inside the training container, detached:
#   nohup bash scripts_qlora_2nodes/sweep_ctx_2nodes.sh > scripts_qlora_2nodes/logs/log_sweep2n.txt 2>&1 &
#
# Env: SEQS="2048 4096 6144 8192 10240 12288 14336 16384" (default), MODE=qlora|lora
#      (default qlora), TAG=$MODE, RUN_TIMEOUT=4500, IDLE_MB=3400, GBS=32,
#      EXTRA_ENV="" (extra "KEY=V KEY=V" for the training script, e.g.
#      "BNB_DEQUANT_PATCH=1 EXPERT_DEQUANT_RECOMPUTE=1 EXPERT_CHUNK_SIZE=16").
#
# If one node fails a length, the peer hangs until HCCL_EXEC_TIMEOUT (1200 s) and then
# fails the same length too — the sweeps stay in lockstep. A timeout kill is local only;
# the peer's run dies on its own via HCCL timeout.
set -u
cd "$(dirname "$0")"
HERE=$(pwd)                                        # scripts_qlora_2nodes
ROOT=$(cd "$HERE/.." && pwd)                       # repo root
SAMPLER=$ROOT/scripts_qlora/perf_ctx_sweep/npu_smi_mb_sampler.py
PEAK=$ROOT/scripts_qlora/perf_ctx_sweep/npu_peak.py

# 2-node rendezvous: MASTER_ADDR = node0's IP reachable from node1; only NODE_RANK
# differs between the nodes (0 on master, 1 on peer). The training script reads the
# same variables from the environment.
MASTER_ADDR=${MASTER_ADDR:-192.168.0.153}
MASTER_PORT=${MASTER_PORT:-29510}
NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR MASTER_PORT

MODE=${MODE:-qlora}
if [ "$MODE" = qlora ]; then
    TRAIN=$HERE/qlora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh
elif [ "$MODE" = lora ]; then
    TRAIN=$HERE/lora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh
else
    echo "ABORT: MODE must be qlora or lora"; exit 1
fi
TAG=${TAG:-$MODE}
GBS=${GBS:-32}
RUN_TIMEOUT=${RUN_TIMEOUT:-4500}
export IDLE_MB=${IDLE_MB:-3400}
SUMMARY=$HERE/summary2n_${TAG}_node$NODE_RANK.txt
RESULTS=$HERE/results2n_${TAG}_node$NODE_RANK.csv
SCRATCH_CKPT=$ROOT/models_ds/perf_sweep2n_unused_node$NODE_RANK

echo "seq,status,rc,iter1_ms,iter2_ms,iter3_ms,s_per_iter_mean23,max_alloc_first_mb,npu_worst_raw_mb,npu_worst_net_mb,$(for d in 0 1 2 3 4 5 6 7; do printf "d${d}_raw_mb,d${d}_net_mb,"; done | sed 's/,$//')" >> "$RESULTS"

wait_free() {
    for i in $(seq 1 60); do
        BUSY=0
        for d in 0 1 2 3 4 5 6 7; do
            R=$(npu-smi info -t usages -i $d | awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2}')
            [ "${R%.*}" -gt 10 ] && BUSY=$((BUSY+1))
        done
        [ $BUSY -eq 0 ] && return 0
        echo "  $BUSY devices busy, waiting ($i)"; sleep 60
    done
    return 1
}

run_one() {
    local SEQ=$1
    local NAME=s${SEQ}_${TAG}
    local LOG=$HERE/log2n_${NAME}_node$NODE_RANK.txt CSV=$HERE/npu_smi_mb2n_${NAME}_node$NODE_RANK.csv
    local DATA=$ROOT/data/data_train_ms_longalpaca/s$SEQ/alpaca
    [ -f "$ROOT/data/data_train_ms_longalpaca/s$SEQ/alpaca_packed_input_ids_document.bin" ] \
        || { echo "ABORT: no dataset s$SEQ under data/data_train_ms_longalpaca"; exit 1; }

    echo "=== $(date '+%F %T') $NAME node$NODE_RANK (mbs 1, gbs $GBS, iters 3) ==="
    wait_free || { echo "ABORT: devices busy"; exit 1; }
    rm -rf "$SCRATCH_CKPT"
    python3 "$SAMPLER" "$CSV" 2 &
    local MPID=$!
    sleep 6
    timeout -s KILL "$RUN_TIMEOUT" env SEQ_LEN=$SEQ MBS=1 GBS=$GBS TRAIN_ITERS=3 \
        DATA_PATH=$DATA CKPT_SAVE_DIR=$SCRATCH_CKPT ${EXTRA_ENV:-} \
        bash "$TRAIN" > "$LOG" 2>&1
    local RC=$?
    # timeout kills only the local launcher; make sure no local training process survives
    pkill -9 -f posttrain_gpt 2>/dev/null
    sleep 20
    kill $MPID 2>/dev/null
    rm -rf "$SCRATCH_CKPT"

    local STATUS
    if grep -aq "iteration        3/" "$LOG"; then STATUS=OK
    elif grep -aqiE "out of memory|OutOfMemoryError" "$LOG"; then
        if grep -aq "iteration        1/" "$LOG"; then STATUS=OOM_TRAIN; else STATUS=OOM_INIT; fi
    else STATUS=FAIL; fi
    { [ $RC -eq 124 ] || [ $RC -eq 137 ]; } && [ "$STATUS" != OK ] && STATUS=${STATUS}_TIMEOUT

    # iteration times (ms), mean of iters 2-3 in seconds
    local TIMES SITER
    TIMES=$(grep -aoE "elapsed time per iteration \(ms\): [0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | head -3)
    SITER=$(echo "$TIMES" | tail -n +2 | awk '{s+=$1; n++} END {if (n>0) printf "%.2f", s/n/1000}')
    # Megatron memory report after iteration 1 (first rank on this node = stage head)
    local ALLOC_FIRST
    ALLOC_FIRST=$(grep -a "(after 1 iterations) memory" "$LOG" | head -1 \
        | grep -oE "max allocated: [0-9.]+" | grep -oE "[0-9.]+")
    # npu-smi peaks (raw and net of the idle baseline), worst local card + per card
    local CARDCSV WORST_RAW WORST_NET
    CARDCSV=$(python3 "$PEAK" "$CSV" --csv)
    WORST_RAW=$(echo "$CARDCSV" | awk -F, '{m=0; for(i=1;i<=NF;i+=2) if($i>m) m=$i; print m}')
    WORST_NET=$(echo "$CARDCSV" | awk -F, '{m=0; for(i=2;i<=NF;i+=2) if($i>m) m=$i; print m}')

    {
        echo "### $NAME node$NODE_RANK  status=$STATUS rc=$RC"
        grep -ahE "^\[Rank [0-9]+\] \(after 1 iterations\) memory" "$LOG" | sort -u
        grep -aoE "iteration +[0-9]+/ +[0-9]+ .*elapsed time per iteration \(ms\): [0-9.]+" "$LOG" \
            | sed -E 's/iteration +([0-9]+)\/.*\(ms\): ([0-9.]+)/  iter \1: \2 ms/'
        grep -aoE "lm loss: [0-9.E+-]+" "$LOG" | head -3 | sed 's/^/  /'
        grep -aiE -m2 "out of memory|OutOfMemoryError|EL0004" "$LOG" | cut -c1-300 | sed 's/^/  /'
        python3 "$PEAK" "$CSV"
        echo
    } | tee -a "$SUMMARY"

    local I1 I2 I3
    I1=$(echo "$TIMES" | sed -n 1p); I2=$(echo "$TIMES" | sed -n 2p); I3=$(echo "$TIMES" | sed -n 3p)
    echo "$SEQ,$STATUS,$RC,${I1:-},${I2:-},${I3:-},${SITER:-},${ALLOC_FIRST:-},${WORST_RAW:-},${WORST_NET:-},${CARDCSV}" >> "$RESULTS"

    SWEEP_STATUS=$STATUS
}

echo "# 2-node context sweep ($TAG, node$NODE_RANK, mode $MODE) on master, $(date '+%F %T'), seqs: ${SEQS:-default}" >> "$SUMMARY"
for SEQ in ${SEQS:-2048 4096 6144 8192 10240 12288 14336 16384}; do
    run_one $SEQ
    if [ "$SWEEP_STATUS" != OK ]; then
        echo "# STOP at s$SEQ ($SWEEP_STATUS) — longer lengths cannot fit, $(date '+%F %T')" | tee -a "$SUMMARY"
        break
    fi
done
echo "# done $(date '+%F %T')" | tee -a "$SUMMARY"
if command -v column >/dev/null; then column -s, -t "$RESULTS"; else cat "$RESULTS"; fi
