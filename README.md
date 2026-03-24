# HRNet + LoveDA 语义分割 → 港口图像分割

基于 **HRNet 官方原生框架**（路线B）的遥感语义分割项目，使用 LoveDA 数据集训练，支持港口图像分割推理与微调。

> **路线B 核心**：不依赖 MMSegmentation，直接使用 HRNet 官方训练框架（YACS 配置系统、自定义训练循环、原生 Dataset 类）。

---

## 与路线A（MMSegmentation）的区别

| 对比项 | 路线A（MMSeg） | 路线B（本项目，HRNet 原生） |
|--------|---------------|---------------------------|
| 框架依赖 | mmcv + mmengine + mmsegmentation | 仅 PyTorch + YACS |
| 配置系统 | Python 配置文件 | YAML（YACS） |
| 学习曲线 | 较陡 | 较平 |
| 灵活性 | 高 | 高 |
| 适合场景 | 快速实验 | 深度定制、轻量部署 |

---

## 完整目录结构

```
├── README.md                                    # 本文件
├── requirements.txt                             # 依赖清单
├── setup_env.sh                                 # 一键环境安装
├── prepare_loveda.py                            # LoveDA 数据准备
├── inference_port.py                            # 港口单张推理 + 可视化
├── inference_batch.py                           # 港口批量推理
├── visualize_results.py                         # 训练曲线可视化
├── tools/
│   ├── _init_paths.py                           # 路径初始化
│   ├── train.py                                 # 训练脚本
│   └── test.py                                  # 测试脚本
├── lib/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── default.py                           # YACS 默认配置
│   │   └── models.py                            # 模型配置
│   ├── datasets/
│   │   ├── __init__.py                          # 数据集注册（含 LoveDA）
│   │   ├── base_dataset.py                      # BaseDataset 基类
│   │   ├── cityscapes.py                        # Cityscapes（参考）
│   │   └── loveda.py                            # LoveDA 数据集（核心新增）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bn_helper.py                         # BatchNorm 辅助
│   │   ├── seg_hrnet.py                         # HRNet 模型
│   │   └── seg_hrnet_ocr.py                     # HRNet-OCR 模型
│   ├── core/
│   │   ├── criterion.py                         # 损失函数
│   │   └── function.py                          # train/validate/testval/test
│   └── utils/
│       ├── __init__.py
│       ├── utils.py                             # 工具函数
│       ├── distributed.py                       # 分布式训练工具
│       └── modelsummary.py                      # 模型摘要
├── experiments/
│   └── loveda/
│       ├── seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml
│       ├── seg_hrnet_w18_train_512x512_sgd_lr1e-2_wd5e-4_bs_16_epoch200.yaml
│       └── seg_hrnet_w48_port_finetune.yaml
├── data/
│   └── loveda/                                  # 数据目录（运行 prepare_loveda.py 生成）
└── scripts/
    ├── train_loveda.sh                          # 单卡训练
    ├── dist_train_loveda.sh                     # 多卡分布式训练
    └── test_loveda.sh                           # 测试评估
```

---

## 一、环境安装

### 方式1：一键安装脚本

```bash
bash setup_env.sh
```

### 方式2：手动安装

```bash
# 创建 conda 环境
conda create -n hrnet_loveda python=3.8 -y
conda activate hrnet_loveda

# 安装 PyTorch（CUDA 11.8）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖（不含任何 mm 系列包）
pip install -r requirements.txt
```

---

## 二、数据准备

### 2.1 下载 LoveDA 数据集

从 [LoveDA 官方仓库](https://github.com/JunjueWang/LoveDA) 下载数据集（约 3GB）。
下载后原始目录结构类似：

```
LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    └── Rural/
```

### 2.2 运行数据准备脚本

```bash
# 整理目录结构 + 生成 lst 文件
python prepare_loveda.py --src /path/to/LoveDA --dst data/loveda

# 验证数据集完整性
python prepare_loveda.py --src /path/to/LoveDA --dst data/loveda --verify

# 如果数据已整理好，只生成 lst 文件
python prepare_loveda.py --skip-organize --dst data/loveda
```

运行后生成：

```
data/loveda/
├── images/
│   ├── train/    # 训练图像
│   ├── val/      # 验证图像
│   └── test/     # 测试图像
├── labels/
│   ├── train/    # 训练标注
│   └── val/      # 验证标注
├── train.lst     # 训练集列表
└── val.lst       # 验证集列表
```

### 2.3 LoveDA 类别说明

| 类别索引 | 类别名 | 说明 | 颜色 |
|---------|--------|------|------|
| 0 | background | 背景 | 黑色 |
| 1 | building | 建筑 | 红色 |
| 2 | road | 道路 | 黄色 |
| 3 | water | 水体 | 蓝色 |
| 4 | barren | 裸地 | 紫色 |
| 5 | forest | 森林 | 绿色 |
| 6 | agriculture | 农田 | 橙色 |

> 标注像素值直接为类别索引（0-6），ignore_label=255。

---

## 三、下载预训练权重

```bash
mkdir -p pretrained_models

# HRNet-W48 ImageNet 预训练权重
wget -O pretrained_models/hrnetv2_w48_imagenet_pretrained.pth \
    https://download.pytorch.org/models/hrnetv2_w48_imagenet_pretrained.pth

# HRNet-W18 ImageNet 预训练权重
wget -O pretrained_models/hrnetv2_w18_imagenet_pretrained.pth \
    https://download.pytorch.org/models/hrnetv2_w18_imagenet_pretrained.pth
```

> 或从 [HRNet 官方 Google Drive](https://github.com/HRNet/HRNet-Image-Classification) 下载预训练权重。

---

## 四、训练

### 4.1 单卡训练（HRNet-W48）

```bash
# 方式1：直接运行
python tools/train.py \
    --cfg experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml

# 方式2：使用脚本
bash scripts/train_loveda.sh

# 使用 HRNet-W18（显存占用更少）
bash scripts/train_loveda.sh \
    experiments/loveda/seg_hrnet_w18_train_512x512_sgd_lr1e-2_wd5e-4_bs_16_epoch200.yaml
```

### 4.2 多卡分布式训练（4 卡）

```bash
# 方式1：直接运行
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py \
    --cfg experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml

# 方式2：使用脚本（默认 4 卡）
bash scripts/dist_train_loveda.sh 4
```

### 4.3 训练输出

训练结果保存至：

```
output/loveda/seg_hrnet/<config_name>/
├── best.pth            # 最佳模型（最高 mIoU）
├── final_state.pth     # 最终模型
├── checkpoint.pth.tar  # 最新 checkpoint（用于断点续训）
└── *.log               # 训练日志

log/loveda/seg_hrnet/<config_name>/
└── events.out.tfevents.*  # TensorBoard 日志
```

### 4.4 查看训练曲线

```bash
# 使用 TensorBoard
tensorboard --logdir log/

# 或使用本项目的可视化脚本
python visualize_results.py --log-dir log/ --output output/curves.png
```

---

## 五、测试评估

```bash
# 方式1：直接运行
python tools/test.py \
    --cfg experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml \
    TEST.MODEL_FILE output/loveda/seg_hrnet/best.pth

# 方式2：使用脚本
bash scripts/test_loveda.sh output/loveda/seg_hrnet/best.pth
```

输出指标：MeanIU（mIoU）、PixelAcc、MeanAcc、各类别 IoU。

---

## 六、港口图像推理

### 6.1 单张图像推理

```bash
# 基础推理
python inference_port.py \
    --cfg experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml \
    --model output/loveda/seg_hrnet/best.pth \
    --image /path/to/port_image.jpg \
    --output output/inference/

# 使用滑窗推理（推荐用于大尺寸图像）
python inference_port.py \
    --model output/loveda/seg_hrnet/best.pth \
    --image /path/to/port_image.jpg \
    --sliding-window

# 使用多尺度推理（精度更高）
python inference_port.py \
    --model output/loveda/seg_hrnet/best.pth \
    --image /path/to/port_image.jpg \
    --multi-scale --scales 0.75 1.0 1.25
```

输出文件：
- `{name}_seg.png`：分割结果（类别索引图）
- `{name}_color.png`：彩色分割图
- `{name}_overlay.png`：叠加图
- `{name}_visualization.png`：3 列对比可视化（含图例）

### 6.2 批量推理

```bash
python inference_batch.py \
    --cfg experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml \
    --model output/loveda/seg_hrnet/best.pth \
    --input-dir /path/to/port_images/ \
    --output-dir output/batch_results/ \
    --save-color \
    --save-overlay
```

---

## 七、港口微调

在 LoveDA 预训练权重基础上，针对港口数据集进行微调：

### 7.1 准备港口数据集

```bash
# 组织港口数据（与 LoveDA 相同格式）
python prepare_loveda.py \
    --src /path/to/port_dataset \
    --dst data/port \
    --verify
```

### 7.2 运行微调

```bash
python tools/train.py \
    --cfg experiments/loveda/seg_hrnet_w48_port_finetune.yaml
```

微调配置特点：
- 学习率：`LR: 0.001`（比从头训练小 10 倍）
- 训练轮数：`END_EPOCH: 50`
- 初始权重：LoveDA 预训练权重

---

## 八、显存需求参考

| 模型 | 输入尺寸 | 批次大小 | 显存占用 |
|------|---------|---------|---------|
| HRNet-W18 | 512×512 | 6 | ~6 GB |
| HRNet-W32 | 512×512 | 4 | ~8 GB |
| HRNet-W48 | 512×512 | 4 | ~12 GB |
| HRNet-W48 | 512×512 | 1 | ~6 GB |

> 显存不足时，可减小 `BATCH_SIZE_PER_GPU` 或使用 HRNet-W18。

---

## 九、常见问题 FAQ

**Q: 训练时出现 `ModuleNotFoundError: No module named 'config'`？**  
A: 请在 `tools/` 目录下运行训练脚本，`_init_paths.py` 会自动将 `lib/` 加入 path。

**Q: `np.int` 报错（DeprecationError）？**  
A: 已修复，新增文件均使用 `int()` 代替 `np.int()`。如果官方代码文件出现此问题，将 `np.int` 改为 `int` 即可。

**Q: 预训练权重加载失败？**  
A: 检查 `PRETRAINED` 配置项中的路径是否正确，权重文件是否完整下载。

**Q: 如何在 CPU 上运行推理？**  
A: 在推理脚本中添加 `--device cpu` 参数。训练脚本修改 `GPUS: ()` 配置（不推荐）。

**Q: LoveDA 标注中像素值范围是什么？**  
A: 标注像素值为 0-6（7 个类别），其中超出范围的像素值会被设为 255（ignore_label）。无需 id→trainId 映射。

**Q: 港口图像效果不好怎么办？**  
A: 建议收集港口标注数据后进行微调（见第七节）。重点关注 Water（水体）和 Building（建筑）这两个港口关键类别的 IoU。

---

## 参考资源

- [HRNet 官方仓库](https://github.com/HRNet/HRNet-Semantic-Segmentation)（HRNet-OCR 分支）
- [LoveDA 数据集](https://github.com/JunjueWang/LoveDA)
- [HRNet 论文](https://arxiv.org/abs/1908.07919)：Deep High-Resolution Representation Learning for Visual Recognition
