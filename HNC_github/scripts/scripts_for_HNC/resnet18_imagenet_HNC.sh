#!/bin/bash
#Submit to GPU

directory=/path/to/save/log
dir_data=/path/to/imagenet
dir_save="${directory}/logs/"
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
EXP_NAME=PARAMETRIC_FP_WAQ_DELTA_XMAX_MobileNet
# MobileNet
MODEL=ResNet_DHP_SHAREL
BATCH=640
LAYER=18
TEMPLATE=ImageNet_resnet18
REG=2e-5 # for l1 norm
T=1e-4
LIMIT=0.004
RATIO=0.016
LR=0.03
CHECKPOINT=${MODEL}_gradually${TEMPLATE}_${EXP_NAME}_B${BATCH}_LR${LR}_Drop_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}${DATE_WITH_TIME}
echo $CHECKPOINT
mkdir -p ${dir_save}${CHECKPOINT}/
python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}" --model ${MODEL} --batch_size ${BATCH} --epochs 150 --decay step-100+step-30-60-90-120 \
--prune_threshold ${T} --depth ${LAYER} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --prune_classifier --linear_percentage 0.45 \
--dir_save ${dir_save} --dir_data ${dir_data} --experiment=${EXP_NAME} --n_GPUs=4  --lr=${LR} --gradually --n_threads=64 --depth ${LAYER} --pretrain_init \
--cfg ../../train_resnet_quant_fp.cfg  --search_lr=0.003 --remain_percentage=0.6 | tee -a ${dir_save}${CHECKPOINT}/alllog.txt
echo $CHECKPOINT





