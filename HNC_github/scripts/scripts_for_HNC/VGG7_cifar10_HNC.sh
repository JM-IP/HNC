#!/bin/bash
#Submit to GPU

directory=/path/to/save/log
dir_data="${directory}/data/"
dir_save="${directory}/logs/"
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
# ResNet20, Ratio=0.5
MODEL=VGG_DHP
LAYER=20
BATCH=256
TEMPLATE=CIFAR10
REG=2e-2 # for l1 norm
T=5e-3
LIMIT=0.00004
RATIO=0.004
LR=0.1
INIT_PRUNE=1
VAR_WEIGHT=0
EXP_NAME=PARAMETRIC_FP_WAQ_DELTA_XMAX
REG_WEIGHT=1
CHECKPOINT=${MODEL}_gradually${TEMPLATE}_${EXP_NAME}_L${LAYER}_B${BATCH}_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}_LR${LR}_BETA${REG_WEIGHT}_INITPRUNE${INIT_PRUNE}_VARV3${VAR_WEIGHT}_TIMESTAMP${DATE_WITH_TIME}
echo $CHECKPOINT
mkdir -p ${dir_save}${CHECKPOINT}
python ../../main_dhp.py --save $CHECKPOINT \
--template "${TEMPLATE}_VGG_DHP" --model ${MODEL} --batch_size ${BATCH} \
--epochs 300 --decay step-50+step-150-225 \
--depth ${LAYER} --prune_threshold ${T} --regularization_factor ${REG} \
--ratio ${RATIO} --stop_limit ${LIMIT} --print_model --n_threads=32 \
--dir_save ${dir_save} --dir_data ${dir_data} --experiment=${EXP_NAME} \
--reg_weight=${REG_WEIGHT} --lr=${LR} --gradually --init_prune=${INIT_PRUNE} --var_weight=${VAR_WEIGHT} \
--cfg ../../train_resnet_quant_fp.cfg   | tee -a ${dir_save}${CHECKPOINT}/alllog.txt


echo $CHECKPOINT

# bash dhp_VGG7_mixed_precision_WAQ_beat_SOTA.sh 8