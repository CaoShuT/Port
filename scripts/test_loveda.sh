#!/bin/bash
# 测试评估启动脚本（HRNet-W48 + LoveDA）
# 使用方法: bash scripts/test_loveda.sh [模型文件] [可选配置文件]

set -e

MODEL_FILE=${1:-"output/loveda/seg_hrnet/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200/best.pth"}
CFG=${2:-"experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml"}

echo "=============================="
echo " 启动测试评估"
echo " 模型文件: ${MODEL_FILE}"
echo " 配置文件: ${CFG}"
echo "=============================="

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python tools/test.py \
    --cfg "${CFG}" \
    TEST.MODEL_FILE "${MODEL_FILE}"
