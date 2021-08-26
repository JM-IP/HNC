#!/bin/bash
#Submit to GPU

directory=/path/to/save/log
dir_data="${directory}/data/"
dir_save="${directory}/logs/"
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
EXP_NAME=PARAMETRIC_FP_WAQ_DELTA_XMAX_MobileNet16bit
# MobileNet
MODEL=MobileNetV2_DHP
BATCH=128
WIDTH=2.0
TEMPLATE=Tiny_ImageNet_MobileNetV2
REG=2e-3 # for l1 norm
T=1e-4
LIMIT=0.004
RATIO=0.08
LR=0.1
CHECKPOINT=${MODEL}init16gradually_${TEMPLATE}_${EXP_NAME}_B${BATCH}_W${WIDTH}_LR${LR}_Drop_Reg${REG}_T${T}_Limit${LIMIT}_Ratio${RATIO}${DATE_WITH_TIME}
echo $CHECKPOINT
python ../../main_dhp.py --save $CHECKPOINT --template "${TEMPLATE}" --model ${MODEL} --batch_size ${BATCH} --epochs 300 --decay step-50-70+step-200-205-210-255 --width_mult ${WIDTH} \
--prune_threshold ${T} --regularization_factor ${REG} --ratio ${RATIO} --stop_limit ${LIMIT} --prune_classifier --linear_percentage 0.45 \
--dir_save ${dir_save} --dir_data ${dir_data} --experiment=${EXP_NAME} --n_GPUs=2  --n_threads=64 \
--cfg ../../train_resnet_quant_fp.cfg --lr=${LR} --gradually
echo $CHECKPOINT


#--save imagenetdebug --lr=0.01 --template "ImageNet_MobileNet" --model MobileNet_DHP --batch_size 64 --epochs 300 --decay step-50+step-150-225 --depth 20 --prune_threshold 5e-3 --regularization_factor 2e-4 --ratio 0.015 --stop_limit 0.02 --print_model --dir_save "~/D/yjm/projects/logs/" --dir_data "/home/yjm/D/yjm/projects/data/imagenet/" --experiment=PARAMETRIC_FP_WAQ_DELTA_XMAX --cfg /home/yjm/D/yjm/Meta_Unifly_l1_notreal/classification/train_resnet_quant_fp.cfg





