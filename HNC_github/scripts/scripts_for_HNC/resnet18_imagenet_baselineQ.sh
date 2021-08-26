#!/bin/bash
#Submit to GPU

directory=/path/to/save/log
dir_data=/path/to/imagenet
dir_save="${directory}/logs/"
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
EXP_NAME=PARAMETRIC_FP_WAQ_DELTA_XMAX_MobileNet
# MobileNet
MODEL=ResNet_PQ
BATCH=700
LAYER=18
TEMPLATE=ImageNet_resnet18
LR=0.3
CHECKPOINT=${MODEL}${TEMPLATE}_${EXP_NAME}_B${BATCH}_LR${LR}_Drop_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}${DATE_WITH_TIME}
echo $CHECKPOINT
mkdir -p ${dir_save}${CHECKPOINT}/
python ../../main.py --save $CHECKPOINT --template "${TEMPLATE}" \
--model ${MODEL} --batch_size ${BATCH} --epochs 100 --decay step-30-60-90-100 \
--depth ${LAYER} --dir_save ${dir_save} --dir_data ${dir_data} --experiment=${EXP_NAME} \
--n_GPUs=4  --lr=${LR} --n_threads=64 --depth ${LAYER} --data_train=ImageNet \
--cfg ../../train_resnet_quant_fp.cfg | tee -a ${dir_save}${CHECKPOINT}/alllog.txt



