export OMP_NUM_THREADS=64
MODEL_PATH=/data/models/DeepSeek-V4-Flash-Base
SAVE_PATH=/data/models/DeepSeek-V4-Flash-Base-w8a8
CONFIG=deepseek_v4_flash_base_w8a8.yaml
mkdir -p $SAVE_PATH

cd $SAVE_PATH
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --device_id 0 1 2 3 4 5 6 7 \
  --model_type DeepSeek-V4-Flash \
  --config ${CONFIG} \
  --trust_remote_code true
