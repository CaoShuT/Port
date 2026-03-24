#!/bin/bash
# 单卡训练 HRNet-W48 on LoveDA
python train.py \
    --config configs/fcn_hr48_loveda.py \
    --work-dir work_dirs/fcn_hr48_loveda \
    --amp

# HRNet-W18 轻量版（显存不足时使用）：
# python train.py \
#     --config configs/fcn_hr18_loveda.py \
#     --work-dir work_dirs/fcn_hr18_loveda \
#     --amp
