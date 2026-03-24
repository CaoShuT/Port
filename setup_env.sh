#!/bin/bash
# -*- coding: utf-8 -*-
# 一键环境安装脚本（不依赖 mmcv/mmseg）

set -e

echo "=============================="
echo " HRNet LoveDA 环境安装脚本"
echo "=============================="

# 创建 conda 环境
conda create -n hrnet_loveda python=3.8 -y
conda activate hrnet_loveda

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 如需 CUDA 11.7，取消注释下行：
# pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 如需 CPU-only，取消注释下行：
# pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖（不包含任何 mm 系列包）
pip install yacs tensorboardX tqdm opencv-python matplotlib pillow numpy

# 验证安装
echo ""
echo "=============================="
echo " 验证安装"
echo "=============================="
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
python -c "import yacs; print(f'YACS 安装成功')"
python -c "import tensorboardX; print(f'TensorboardX 安装成功')"
python -c "import cv2; print(f'OpenCV 版本: {cv2.__version__}')"
python -c "import PIL; print(f'Pillow 版本: {PIL.__version__}')"

echo ""
echo "=============================="
echo " 环境安装完成！"
echo " 激活环境: conda activate hrnet_loveda"
echo "=============================="
