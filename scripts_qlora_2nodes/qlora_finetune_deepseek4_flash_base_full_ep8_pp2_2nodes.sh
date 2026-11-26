#!/bin/bash
# Full-model QLoRA training on TWO nodes (16 cards, EP8 / PP2 / TP1), master branch
# (dev/qlora_support). 2-node counterpart of
# scripts_qlora/qlora_finetune_deepseek4_flash_base_full_ep4_pp2_smoke.sh, adapted from
# the old fork's lora_finetune_deepseek4_flash_0731_ep8_pp2_full_qlora_2nodes_gsm8k_comp.sh
# with master's argument set (docs/ds_args.md):
#   * --position-embedding-type deepseek4, --sliding-window-size (no g2 flags)
#   * fused-op flags on (validated on the single-node full-model runs)
#   * ENABLE_BNB_PATCH / QLORA_EXPERT_CHUNK are real args now (QLORA_OPT_ARGS below)
#
# LAUNCH ON BOTH NODES with the same parameters; only NODE_RANK differs.
# Node0 holds PP stage 0 (ranks 0-7), node1 stage 1 (ranks 8-15); stage 0 is the heavier
# one and sets the memory limit. Start node1 first, node0 last (rendezvous server).
#   node0:  NODE_RANK=0 bash scripts_qlora_2nodes/qlora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh
#   node1:  NODE_RANK=1 bash scripts_qlora_2nodes/qlora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh
# Detached inside the training container:
#   nohup bash scripts_qlora_2nodes/qlora_finetune_deepseek4_flash_base_full_ep8_pp2_2nodes.sh > scripts_qlora_2nodes/logs/log_train_qlora_2nodes.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/scripts_qlora_2nodes/logs"
echo "Start time:"
date

export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export ACL_DEVICE_SYNC_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export NPU_VISIBLE_DEVICES=$DEVICES

# 2-node rendezvous. MASTER_ADDR = node0's IP reachable from node1 (HCCL/RoCE
# interface). Both nodes run the SAME script with the same parameters; only NODE_RANK
# differs (0 on the master, 1 on the peer). Override via environment or edit here.
MASTER_ADDR=${MASTER_ADDR:-192.168.0.153}
MASTER_PORT=${MASTER_PORT:-29510}
NODE_RANK=${NODE_RANK:-0}

NPUS_PER_NODE=8
NNODES=2

# Writable shared cache for MindSpeed's PyTorch JIT extensions.
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$ROOT/.torch_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"
find "$TORCH_EXTENSIONS_DIR" -maxdepth 1 -type f -name "*lock" -print -delete 2> /dev/null

###### TRAINING ARGS ######

TRAIN_ITERS=${TRAIN_ITERS:-5}
MBS=${MBS:-1}
GBS=${GBS:-32}
SEQ_LEN=${SEQ_LEN:-2048}
# The rope table only needs to cover SEQ_LEN, with a minimum of 4096 rows.
ROPE_TABLE_LEN=$(( SEQ_LEN > 4096 ? SEQ_LEN : 4096 ))

###### PATHS ######

REPO=${REPO:-$ROOT/MindSpeed-LLM}
CKPT_LOAD_DIR=${CKPT_LOAD_DIR:-$ROOT/models_ds/deepseek4_flash_base_full_ep8_pp2_nf4_mcore_comp}
CKPT_SAVE_DIR=${CKPT_SAVE_DIR:-$ROOT/models_ds/deepseek4_flash_base_full_ep8_pp2_qlora_2nodes_smoke}
DATA_PATH=${DATA_PATH:-$ROOT/data/data_train_ms_longalpaca/s2048/alpaca}
TOKENIZER_PATH=${TOKENIZER_PATH:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-bf16}

# Dynamic-length batching: pad batches only to a multiple of 128 instead of the full
# seq-length (run D's setup for the variable-length distill data). Off by default; the
# default smoke data is pre-padded, where this is a no-op anyway. When enabled, the save
# dir gets a _nopad suffix so it cannot clobber a padded run's adapter.
NOPAD=${NOPAD:-0}
PAD_ARGS=""
if [ "$NOPAD" = "1" ]; then
    PAD_ARGS="--no-pad-to-seq-lengths --pad-to-multiple-of 128"
    CKPT_SAVE_DIR="${CKPT_SAVE_DIR}_nopad"
fi

if [ -e "$CKPT_SAVE_DIR" ]; then
    echo "ABORT: $CKPT_SAVE_DIR exists"; exit 1;
fi

##### MINDSPEED ARGS ######

TP=1
PP=2
EP=8
CP=1
CP_TYPE='ulysses_cp_algo'

NUM_LAYERS=44    # 43 + 1 mtp
NOOP_LAYERS=43
N_HASH_LAYERS=${N_HASH_LAYERS:-3}   # fits the 22-layer PP stages

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DSA_ARGS="
    --enable-dsa-indexer \
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 512 \
    --enable-mhc \
    --use-fused-mhc \
    --hc-mult 4 \
    --kv-compress \
    --use-fused-lightning-indexer-loss \
    --use-fused-lightning-indexer \
    --use-sparse-flash-attn \
    --indexer-loss-coeff 1.0 \
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 512 \
    --q-lora-rank 1024 \
    --o-lora-rank 1024 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-fa-without-pad \
"

CA_ARGS="
    --o-groups 8 \
    --sliding-window-size 128 \
    --original-seq-len 65536 \
    --rope-factor 16 \
    --compress-rope-theta 160000.0 \
    --max-batch-size 4 \
    --compress-ratios 0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 0 \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq 1 \
    --first-k-dense-replace -1 \
    --num-experts 256 \
    --moe-router-topk 6 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type none \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 1 \
    --moe-router-topk-scaling-factor 1.5 \
    --seq-aux \
    --moe-aux-loss-coeff 0.001 \
    --moe-router-score-function sqrtsoftplus \
    --moe-router-enable-expert-bias \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-dtype fp32 \
    --n-hash-layers ${N_HASH_LAYERS} \
    --moe-permute-fusion \
"

MEM_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --swap-optimizer \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings ${ROPE_TABLE_LEN} \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --noop-layers ${NOOP_LAYERS} \
    --transformer-impl local \
    --spec mindspeed_llm.tasks.models.spec.deepseek4_spec layer_spec \
    --manual-gc \
    --manual-gc-interval 50 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 3e-4 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type deepseek4 \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-5 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --no-shared-storage \
    --no-gradient-accumulation-fusion \
    --seed 42
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval 5000 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --ckpt-format torch
"

FINETUNE_ARGS="
    --finetune \
    --qlora \
    --stage sft \
    --is-instruction-dataset \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type deepseek4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_fc1 linear_fc2 \
    --lora-ckpt-filter
"

# QLoRA memory optimizations (all on dev/qlora_support, default off).
# EXPERT_CHUNK_SIZE needs EXPERT_DEQUANT_RECOMPUTE=1. Validated all-on single-node
# config: BNB_DEQUANT_PATCH=1 EXPERT_DEQUANT_RECOMPUTE=1 EXPERT_CHUNK_SIZE=16.
BNB_DEQUANT_PATCH=${BNB_DEQUANT_PATCH:-0}
EXPERT_DEQUANT_RECOMPUTE=${EXPERT_DEQUANT_RECOMPUTE:-0}
EXPERT_CHUNK_SIZE=${EXPERT_CHUNK_SIZE:-0}
QLORA_OPT_ARGS=""
if [ "$BNB_DEQUANT_PATCH" = 1 ]; then
    QLORA_OPT_ARGS="$QLORA_OPT_ARGS --bnb-dequant-patch"
fi
if [ "$EXPERT_DEQUANT_RECOMPUTE" = 1 ]; then
    QLORA_OPT_ARGS="$QLORA_OPT_ARGS --expert-dequant-recompute"
fi
if [ "$EXPERT_CHUNK_SIZE" -gt 0 ]; then
    QLORA_OPT_ARGS="$QLORA_OPT_ARGS --expert-chunk-size $EXPERT_CHUNK_SIZE"
fi

python -m torch.distributed.launch $DISTRIBUTED_ARGS $REPO/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $DSA_ARGS \
    $CA_ARGS \
    $MEM_ARGS \
    $FINETUNE_ARGS \
    $QLORA_OPT_ARGS \
    $PAD_ARGS
