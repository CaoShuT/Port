#!/bin/bash
# 多卡分布式训练
CONFIG=${1:-"configs/fcn_hr48_loveda.py"}
GPUS=${2:-4}
WORK_DIR=${3:-"work_dirs/fcn_hr48_loveda"}
PORT=${PORT:-29500}

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG \
    --work-dir $WORK_DIR \
    --launcher pytorch \
    --amp

# 使用示例：
# bash scripts/dist_train.sh configs/fcn_hr48_loveda.py 4 work_dirs/fcn_hr48_loveda
# bash scripts/dist_train.sh configs/fcn_hr48_port_finetune.py 2 work_dirs/fcn_hr48_port
