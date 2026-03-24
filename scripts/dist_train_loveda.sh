#!/bin/bash
# 多卡分布式训练启动脚本（HRNet-W48 + LoveDA）
# 使用方法: bash scripts/dist_train_loveda.sh [GPU数量] [可选配置文件]

set -e

NPROC=${1:-4}
CFG=${2:-"experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml"}

echo "=============================="
echo " 启动分布式训练"
echo " GPU 数量: ${NPROC}"
echo " 配置文件: ${CFG}"
echo "=============================="

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python -m torch.distributed.launch \
    --nproc_per_node=${NPROC} \
    tools/train.py \
    --cfg "${CFG}"
