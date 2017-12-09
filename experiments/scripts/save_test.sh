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

DEV=$1
DEV_ID=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/faster_rcnn_end2end_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

  #--weights data/pretrain_model/VGG_imagenet.npy \
time python -m pdb ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_5.ckpt \
  --imdb voc_2007_trainval \
  --iters 10 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}

