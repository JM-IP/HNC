#!/bin/bash
#Submit to GPU

directory=/path/to/save/log&data
dir_data="${directory}/data/"
dir_save="${directory}/logs/"
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
# ResNet20, Ratio=0.5
MODEL=ResNet_DHP_SHARE
LAYER=20
BATCH=64
TEMPLATE=CIFAR10
REG=2e-2 # for l1 norm
T=1e-4
LIMIT=0.00004
RATIO=0.015
LR=0.03
INIT_PRUNE=1
VAR_WEIGHT=0
EXP_NAME=PARAMETRIC_FP_WAQ_DELTA_XMAX
REG_WEIGHT=1
CHECKPOINT=${MODEL}_gradually${TEMPLATE}_${EXP_NAME}_L${LAYER}_B${BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}_LR${LR}_BETA${REG_WEIGHT}_INITPRUNE${INIT_PRUNE}_TIMESTAMP${DATE_WITH_TIME}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=0 python ../../main_dhp.py --save $CHECKPOINT \
--template "${TEMPLATE}_ResNet" --model ${MODEL} --batch_size ${BATCH} \
--epochs 300 --decay step-50+step-150-225 \
--depth ${LAYER} --prune_threshold ${T} --regularization_factor ${REG} \
--ratio ${RATIO} --stop_limit ${LIMIT} --print_model \
--dir_save ${dir_save} --dir_data ${dir_data} --experiment=${EXP_NAME} \
--cfg ../../train_resnet_quant_fp.cfg \
--reg_weight=${REG_WEIGHT} --lr=${LR} --gradually --init_prune=${INIT_PRUNE}
echo $CHECKPOINT