#!/bin/bash
# Create the training or inference container on an Ascend host.
#   STAGE=train bash create_container.sh   # th_va30_qlora_train from the va0.30.0rc2 image
#   STAGE=infer bash create_container.sh   # th_va30_qlora_infer from the va0.30.0rc2 image
# Afterwards run installation/prepare_train_container.sh (inside train) or
# installation/prepare_eval_container.sh + rebuild_va.sh (inside infer).
set -euo pipefail

IMAGE=${IMAGE:-0faf08fb90dd}
STAGE=${STAGE:-}
case "$STAGE" in
  train) NAME=th_va30_qlora_train;;
  infer) NAME=th_va30_qlora_infer;;
  *) echo "Usage: STAGE=train|infer bash $0" >&2; exit 1 ;;
esac
NAME=${NAME_OVERRIDE:-$NAME}

# 322dc508d91d va25.rc1
# 0faf08fb90dd va0.30.0.rc2

docker run -itd --name "$NAME" \
--shm-size 50g \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /data/models:/data/models \
-v /data/lab-shared:/data/lab-shared \
--user root \
--entrypoint /bin/bash \
--net=host \
--privileged \
"$IMAGE"

echo "Created container $NAME"