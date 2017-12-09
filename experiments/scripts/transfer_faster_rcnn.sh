#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=${1-gpu}
DEV_ID=${2-1}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/faster_rcnn_end2end_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_transfer_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights data/pretrain_model/VGG_imagenet.npy \
  --source_imdb transfer_source_trainval \
  --source_data_path data/driving_50k \
  --target_imdb transfer_target_trainval \
  --target_data_path data/KITTI \
  --iters 70000 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}
