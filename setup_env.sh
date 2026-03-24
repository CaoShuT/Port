#!/bin/bash
# 一键环境安装脚本：HRNet + LoveDA 语义分割项目

set -e

echo "========================================"
echo "  HRNet + LoveDA 环境安装脚本"
echo "========================================"

# 创建 conda 环境
conda create -n hrnet_loveda python=3.8 -y
# 激活环境（兼容非交互式 shell）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrnet_loveda

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.7 选项（取消注释使用）：
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
#     --index-url https://download.pytorch.org/whl/cu117

# CPU 选项（取消注释使用）：
# pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# 通过 openmim 安装 mmengine 和 mmcv
pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0"

# 安装 mmsegmentation
pip install "mmsegmentation>=1.0.0"

# 安装其他依赖
pip install numpy pillow matplotlib tqdm opencv-python

echo "========================================"
echo "  验证安装"
echo "========================================"

python -c "
import torch
import mmengine
import mmseg
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
print(f'MMEngine 版本: {mmengine.__version__}')
print(f'MMSeg 版本: {mmseg.__version__}')
print('安装验证成功！')
"

echo "========================================"
echo "  环境安装完成！"
echo "  激活命令: conda activate hrnet_loveda"
echo "========================================"
