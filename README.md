# HRNet + LoveDA 语义分割 → 港口图像分割

基于 **HRNet** 和 **LoveDA** 数据集的遥感语义分割项目，通过在 LoveDA 大规模遥感语义分割数据集上预训练，再微调到港口遥感图像场景，实现高精度港口语义分割。

---

## 项目简介

本项目使用 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 框架，实现了完整的 HRNet 语义分割训练流程：

1. **预训练**：在 LoveDA 数据集（7 类地物）上训练 HRNet-W48
2. **微调**：在港口遥感图像数据集上进行迁移学习微调
3. **推理**：支持单张和批量港口图像的语义分割推理

---

## 项目文件结构

```
Port/
├── README.md                          # 使用说明（本文件）
├── requirements.txt                   # 依赖包列表
├── setup_env.sh                       # 一键环境安装脚本
├── prepare_loveda.py                  # 数据集准备与格式转换
├── train.py                           # 训练脚本
├── test.py                            # 测试评估脚本
├── inference_port.py                  # 港口图像单张推理
├── inference_batch.py                 # 批量推理脚本
├── visualize_results.py               # 训练结果可视化（loss/mIoU曲线）
├── configs/
│   ├── _base_/
│   │   ├── models/fcn_hr18.py        # HRNet-W18 基础模型配置
│   │   ├── datasets/loveda.py        # LoveDA 数据集配置
│   │   ├── schedules/schedule_80k.py # 80k 迭代训练策略
│   │   └── default_runtime.py        # 默认运行时配置
│   ├── fcn_hr18_loveda.py            # HRNet-W18 + LoveDA
│   ├── fcn_hr48_loveda.py            # HRNet-W48 + LoveDA（推荐）
│   └── fcn_hr48_port_finetune.py     # HRNet-W48 港口微调配置
└── scripts/
    ├── train.sh                       # 单卡训练启动
    ├── dist_train.sh                  # 多卡分布式训练启动
    └── test.sh                        # 测试启动
```

---

## 快速开始

### 1. 环境安装

```bash
# 方法一：一键安装（推荐）
bash setup_env.sh

# 方法二：手动安装
conda create -n hrnet_loveda python=3.8 -y
conda activate hrnet_loveda
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install -r requirements.txt
```

### 2. 数据准备

**LoveDA 数据集**（从 [LoveDA 官方](https://github.com/JunjueWang/LoveDA) 下载后执行）：

```bash
# 将 LoveDA 原始格式转换为 MMSeg 格式
python prepare_loveda.py --src /path/to/LoveDA --dst data/loveDA

# 验证数据集完整性
python prepare_loveda.py --verify data/loveDA
```

**港口自定义数据集**（用于微调）：

```bash
python prepare_loveda.py \
    --port-img /path/to/port/images \
    --port-mask /path/to/port/masks \
    --port-output data/port
```

转换后的目录结构：

```
data/loveDA/
├── img_dir/
│   ├── train/    # Urban_*.png, Rural_*.png
│   └── val/
└── ann_dir/
    ├── train/
    └── val/
```

### 3. 训练

```bash
# 单卡训练 HRNet-W48（推荐）
bash scripts/train.sh

# 或直接运行
python train.py \
    --config configs/fcn_hr48_loveda.py \
    --work-dir work_dirs/fcn_hr48_loveda \
    --amp

# 多卡分布式训练（4 卡）
bash scripts/dist_train.sh configs/fcn_hr48_loveda.py 4

# 轻量版 HRNet-W18（显存不足时）
python train.py \
    --config configs/fcn_hr18_loveda.py \
    --work-dir work_dirs/fcn_hr18_loveda \
    --amp
```

### 4. 测试评估

```bash
# 基础测试
bash scripts/test.sh

# 或直接运行，保存可视化结果
python test.py \
    --config configs/fcn_hr48_loveda.py \
    --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
    --show-dir work_dirs/fcn_hr48_loveda/vis_results

# TTA（测试时增强，通常可提升 1-2% mIoU）
python test.py \
    --config configs/fcn_hr48_loveda.py \
    --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
    --tta
```

### 5. 港口图像推理

**单张推理：**

```bash
python inference_port.py \
    --config configs/fcn_hr48_loveda.py \
    --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
    --img port_images/test_001.png \
    --output results/inference/
```

**批量推理：**

```bash
python inference_batch.py \
    --config configs/fcn_hr48_loveda.py \
    --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
    --img-dir port_images/ \
    --output-dir results/batch/
```

### 6. 港口数据集微调

在 LoveDA 预训练权重基础上微调到港口数据集：

```bash
# 确保已完成 LoveDA 训练并获得 iter_80000.pth
python train.py \
    --config configs/fcn_hr48_port_finetune.py \
    --work-dir work_dirs/fcn_hr48_port
```

如需自定义港口类别，编辑 `configs/fcn_hr48_port_finetune.py` 中的 `PORT_CLASSES` 和 `PORT_PALETTE`。

### 7. 可视化训练曲线

```bash
python visualize_results.py \
    --log work_dirs/fcn_hr48_loveda/时间戳/vis_data/scalars.json \
    --output results/training_curves.png
```

---

## LoveDA 类别说明

| 索引 | 类别名称 | 颜色 | 与港口场景的关联 |
|------|----------|------|-----------------|
| 0 | background（背景） | 黑色 | 无关区域 |
| 1 | building（建筑） | 红色 | 港口仓库、码头建筑 |
| 2 | road（道路） | 黄色 | 港口道路、堆场通道 |
| 3 | water（水体） | 蓝色 | 港口水域、航道 |
| 4 | barren（裸地） | 紫色 | 空置堆场 |
| 5 | forest（森林） | 绿色 | 绿化区域 |
| 6 | agriculture（农田） | 橙色 | 港口周边农田 |

> 💡 港口场景核心类别：**building**（仓库）、**road**（道路）、**water**（港池）
> 如需识别船舶、集装箱、吊机等港口特有目标，需要自定义标注并微调。

---

## 显存参考

| 模型 | 批次大小 | 裁剪尺寸 | 显存需求 |
|------|---------|---------|---------|
| HRNet-W18 | 4 | 512×512 | ~8 GB |
| HRNet-W18 | 2 | 512×512 | ~5 GB |
| HRNet-W48 | 4 | 512×512 | ~16 GB |
| HRNet-W48 | 2 | 512×512 | ~10 GB |
| HRNet-W48 | 1 | 512×512 | ~6 GB |

> 显存不足时，可减小 `batch_size` 或将 `crop_size` 改为 `(256, 256)`。

---

## 参考链接

- **MMSegmentation**：https://github.com/open-mmlab/mmsegmentation
- **HRNet 官方**：https://github.com/HRNet/HRNet-Semantic-Segmentation
- **LoveDA 数据集**：https://github.com/JunjueWang/LoveDA
- **MMEngine**：https://github.com/open-mmlab/mmengine
