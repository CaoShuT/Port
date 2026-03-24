#!/bin/bash
# 单卡训练启动脚本（HRNet-W48 + LoveDA）
# 使用方法: bash scripts/train_loveda.sh [可选配置文件]

set -e

CFG=${1:-"experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml"}

echo "=============================="
echo " 启动单卡训练"
echo " 配置文件: ${CFG}"
echo "=============================="

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python tools/train.py --cfg "${CFG}"
