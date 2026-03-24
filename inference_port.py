#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_port.py
港口单张图像推理与可视化脚本
不依赖 MMSeg API，直接使用 PyTorch 加载 HRNet 模型进行推理。
支持滑窗推理和多尺度推理。
输出: 叠加图、纯分割图、各类像素统计，3 列对比可视化。
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# 将 lib 目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from config import config, update_config


# LoveDA 7 类别定义
CLASS_NAMES = [
    'background',   # 0
    'building',     # 1
    'road',         # 2
    'water',        # 3
    'barren',       # 4
    'forest',       # 5
    'agriculture',  # 6
]

# 类别颜色（RGB）
CLASS_COLORS = np.array([
    [0,   0,   0],    # background - 黑色
    [255,  0,   0],   # building   - 红色
    [255, 255,  0],   # road       - 黄色
    [0,   0, 255],    # water      - 蓝色
    [159, 129, 183],  # barren     - 紫色
    [0,  255,   0],   # forest     - 绿色
    [255, 195, 128],  # agriculture- 橙色
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='港口图像语义分割推理')
    parser.add_argument('--cfg', type=str,
                        default='experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml',
                        help='YACS 配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default='output/inference',
                        help='输出目录')
    parser.add_argument('--sliding-window', action='store_true',
                        help='使用滑窗推理（适合大尺寸图像）')
    parser.add_argument('--multi-scale', action='store_true',
                        help='使用多尺度推理')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='多尺度推理尺度列表，例如: 0.75 1.0 1.25')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备: cuda 或 cpu')
    parser.add_argument('opts',
                        help='通过命令行覆盖配置项',
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_model(cfg, model_path, device):
    """加载 HRNet 模型和权重"""
    import models
    model = eval('models.' + cfg.MODEL.NAME + '.get_seg_model')(cfg)
    model.eval()

    # 加载权重
    print(f'=> 加载模型权重: {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 处理 DataParallel / FullModel 包装的 key
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去除 'module.model.' 或 'module.' 前缀
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 检查 key 匹配情况并记录警告
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys())
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys
    if missing_keys:
        print(f'[WARN] 以下 key 未在权重文件中找到 ({len(missing_keys)} 个): '
              f'{list(missing_keys)[:5]}...' if len(missing_keys) > 5
              else f'[WARN] 缺少 key: {missing_keys}')
    if unexpected_keys:
        print(f'[WARN] 权重文件中存在多余的 key ({len(unexpected_keys)} 个): '
              f'{list(unexpected_keys)[:5]}...' if len(unexpected_keys) > 5
              else f'[WARN] 多余 key: {unexpected_keys}')

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    print(f'=> 模型加载成功，共 {sum(p.numel() for p in model.parameters()):,} 个参数')
    return model


def preprocess_image(image_path, mean=None, std=None):
    """读取并预处理图像"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'无法读取图像: {image_path}')
    original = image.copy()
    h, w = image.shape[:2]

    # BGR -> RGB, /255, 归一化
    image = image.astype(np.float32)[:, :, ::-1]
    image /= 255.0
    image -= np.array(mean, dtype=np.float32)
    image /= np.array(std, dtype=np.float32)

    # HWC -> CHW -> NCHW
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).float()

    return image, original, (h, w)


def sliding_window_inference(model, image_tensor, crop_size, device,
                              num_classes, stride_rate=2.0/3.0,
                              align_corners=True):
    """滑窗推理（适合大于 crop_size 的图像）"""
    _, _, h, w = image_tensor.shape
    crop_h, crop_w = crop_size

    # 如果图像小于 crop_size，直接推理
    if h <= crop_h and w <= crop_w:
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            pred = model(image_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return F.interpolate(pred, size=(h, w), mode='bilinear',
                             align_corners=align_corners)

    stride_h = int(crop_h * stride_rate)
    stride_w = int(crop_w * stride_rate)
    full_probs = torch.zeros((1, num_classes, h, w)).to(device)
    count_mat = torch.zeros((1, 1, h, w)).to(device)

    h_grids = max(h - crop_h + stride_h - 1, 0) // stride_h + 1
    w_grids = max(w - crop_w + stride_w - 1, 0) // stride_w + 1

    for ph in range(h_grids):
        for pw in range(w_grids):
            y1 = ph * stride_h
            x1 = pw * stride_w
            y2 = min(y1 + crop_h, h)
            x2 = min(x1 + crop_w, w)
            y1 = max(y2 - crop_h, 0)
            x1 = max(x2 - crop_w, 0)

            crop_img = image_tensor[:, :, y1:y2, x1:x2]
            # 填充到 crop_size
            pad_h = max(crop_h - crop_img.shape[2], 0)
            pad_w = max(crop_w - crop_img.shape[3], 0)
            crop_img = F.pad(crop_img, (0, pad_w, 0, pad_h))
            crop_img = crop_img.to(device)

            with torch.no_grad():
                pred = model(crop_img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            # 上采样到 crop_size
            pred = F.interpolate(pred, size=(crop_h, crop_w), mode='bilinear',
                                 align_corners=align_corners)
            full_probs[:, :, y1:y2, x1:x2] += pred[:, :, :y2 - y1, :x2 - x1]
            count_mat[:, :, y1:y2, x1:x2] += 1

    return full_probs / count_mat


def multi_scale_inference(model, image_tensor, scales, crop_size, device,
                           num_classes, align_corners=True):
    """多尺度推理"""
    _, _, h, w = image_tensor.shape
    full_probs = torch.zeros((1, num_classes, h, w)).to(device)

    for scale in scales:
        new_h = int(h * scale)
        new_w = int(w * scale)
        scaled_img = F.interpolate(image_tensor, size=(new_h, new_w),
                                   mode='bilinear', align_corners=align_corners)
        pred = sliding_window_inference(model, scaled_img, crop_size, device,
                                        num_classes, align_corners=align_corners)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear',
                             align_corners=align_corners)
        full_probs += pred

    return full_probs / len(scales)


def colorize_prediction(pred_map, colors=CLASS_COLORS):
    """将预测类别图转换为彩色图像"""
    color_map = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.uint8)
    for cls_id, color in enumerate(colors):
        mask = pred_map == cls_id
        color_map[mask] = color
    return color_map


def visualize_results(original_bgr, color_seg, pred_map, output_path, alpha=0.5):
    """
    3 列对比可视化:
    左: 原始图像
    中: 分割结果
    右: 叠加图
    """
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay = (original_rgb.astype(np.float32) * (1 - alpha) +
               color_seg.astype(np.float32) * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('港口图像语义分割结果', fontsize=14, fontweight='bold')

    axes[0].imshow(original_rgb)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(color_seg)
    axes[1].set_title('分割结果')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('叠加图')
    axes[2].axis('off')

    # 添加图例
    patches = []
    unique_classes = np.unique(pred_map)
    for cls_id in unique_classes:
        if cls_id < len(CLASS_NAMES):
            color = CLASS_COLORS[cls_id] / 255.0
            pixel_count = np.sum(pred_map == cls_id)
            pct = pixel_count / pred_map.size * 100
            label = f'{CLASS_NAMES[cls_id]} ({pct:.1f}%)'
            patches.append(mpatches.Patch(color=color, label=label))

    fig.legend(handles=patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'=> 可视化结果已保存: {output_path}')


def print_statistics(pred_map, total_pixels):
    """打印各类像素统计"""
    print('\n============================')
    print(' 各类别像素统计')
    print('============================')
    print(f'{"类别":<15} {"像素数":>12} {"占比":>8}')
    print('-' * 38)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = int(np.sum(pred_map == cls_id))
        pct = count / total_pixels * 100
        if count > 0:
            print(f'{cls_name:<15} {count:>12,} {pct:>7.2f}%')
    print('============================\n')


def main():
    args = parse_args()

    # 加载配置
    class FakeArgs:
        cfg = args.cfg
        opts = args.opts

    update_config(config, FakeArgs())

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'=> 使用设备: {device}')

    # 输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    model = load_model(config, args.model, device)

    # 预处理图像
    print(f'=> 读取图像: {args.image}')
    image_tensor, original_bgr, (h, w) = preprocess_image(args.image)
    print(f'   图像尺寸: {h} x {w}')

    # 推理
    crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    num_classes = config.DATASET.NUM_CLASSES

    print('=> 开始推理...')
    if args.multi_scale:
        print(f'   多尺度推理，尺度: {args.scales}')
        probs = multi_scale_inference(
            model, image_tensor, args.scales, crop_size, device,
            num_classes, config.MODEL.ALIGN_CORNERS)
    elif args.sliding_window:
        print(f'   滑窗推理，窗口尺寸: {crop_size}')
        probs = sliding_window_inference(
            model, image_tensor, crop_size, device,
            num_classes, align_corners=config.MODEL.ALIGN_CORNERS)
    else:
        # 直接推理（resize 到 crop_size）
        img_resized = F.interpolate(image_tensor, size=crop_size,
                                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        img_resized = img_resized.to(device)
        with torch.no_grad():
            pred = model(img_resized)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        probs = F.interpolate(pred, size=(h, w), mode='bilinear',
                              align_corners=config.MODEL.ALIGN_CORNERS)

    # 后处理
    pred_map = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    color_seg = colorize_prediction(pred_map)

    # 打印统计
    print_statistics(pred_map, h * w)

    # 保存结果
    img_name = os.path.splitext(os.path.basename(args.image))[0]

    # 保存纯分割图（类别索引）
    seg_path = os.path.join(args.output, f'{img_name}_seg.png')
    Image.fromarray(pred_map).save(seg_path)
    print(f'=> 分割结果已保存: {seg_path}')

    # 保存彩色分割图
    color_path = os.path.join(args.output, f'{img_name}_color.png')
    Image.fromarray(color_seg).save(color_path)
    print(f'=> 彩色分割图已保存: {color_path}')

    # 保存叠加图
    overlay = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay = (overlay.astype(np.float32) * 0.5 +
               color_seg.astype(np.float32) * 0.5).astype(np.uint8)
    overlay_path = os.path.join(args.output, f'{img_name}_overlay.png')
    Image.fromarray(overlay).save(overlay_path)
    print(f'=> 叠加图已保存: {overlay_path}')

    # 保存 3 列对比可视化
    vis_path = os.path.join(args.output, f'{img_name}_visualization.png')
    visualize_results(original_bgr, color_seg, pred_map, vis_path)

    print(f'\n推理完成！所有结果保存至: {args.output}')


if __name__ == '__main__':
    main()
