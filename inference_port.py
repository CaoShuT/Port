#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港口图像单张推理脚本：对单张港口遥感图像进行语义分割推理，
生成叠加图、纯分割图，并打印各类别像素占比统计。

使用示例：
    python inference_port.py \
        --config configs/fcn_hr48_loveda.py \
        --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
        --img port_images/test_001.png \
        --output results/
"""

import argparse
import os

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# LoveDA 7 类定义
LOVEDA_CLASSES = (
    'background', 'building', 'road', 'water',
    'barren', 'forest', 'agriculture'
)

# LoveDA 调色板（RGB）
LOVEDA_PALETTE = np.array([
    [0, 0, 0],        # background - 黑色
    [255, 0, 0],      # building - 红色
    [255, 255, 0],    # road - 黄色
    [0, 0, 255],      # water - 蓝色
    [159, 129, 183],  # barren - 紫色
    [0, 255, 0],      # forest - 绿色
    [255, 195, 128],  # agriculture - 橙色
], dtype=np.uint8)


def inference_single(model, img_path, output_dir, opacity=0.5):
    """
    对单张图像进行语义分割推理。

    Args:
        model:      MMSeg 推理模型
        img_path:   输入图像路径
        output_dir: 输出目录
        opacity:    分割图叠加透明度（0-1）

    Returns:
        seg_map: 分割结果（H x W，类别索引）
    """
    from mmseg.apis import inference_model

    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # 推理
    result = inference_model(model, img_path)
    seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

    # 生成彩色分割图
    color_seg = LOVEDA_PALETTE[seg_map]

    # 读取原始图像（BGR -> RGB）
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 调整分割图尺寸与原图一致
    if color_seg.shape[:2] != img_rgb.shape[:2]:
        color_seg = cv2.resize(
            color_seg, (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        seg_map = cv2.resize(
            seg_map.astype(np.uint8),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    # 生成叠加图
    overlay = (img_rgb * (1 - opacity) + color_seg * opacity).astype(np.uint8)

    # 保存结果
    overlay_path = os.path.join(output_dir, f'{img_name}_overlay.png')
    seg_path = os.path.join(output_dir, f'{img_name}_seg.png')

    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(color_seg).save(seg_path)

    # 打印各类像素占比
    total_pixels = seg_map.size
    print(f"\n图像：{img_path}")
    print(f"分割结果尺寸：{seg_map.shape}")
    print(f"{'类别':<15} {'像素数':>10} {'占比':>8}")
    print('-' * 36)
    for cls_id, cls_name in enumerate(LOVEDA_CLASSES):
        count = np.sum(seg_map == cls_id)
        ratio = count / total_pixels * 100
        if count > 0:
            print(f"{cls_id}. {cls_name:<12} {count:>10} {ratio:>7.2f}%")

    print(f"\n叠加图已保存：{overlay_path}")
    print(f"分割图已保存：{seg_path}")

    return seg_map


def visualize_with_legend(img_path, seg_map, output_path, opacity=0.5):
    """
    使用 matplotlib 生成 3 列对比图（原图/分割图/叠加图）加图例。

    Args:
        img_path:    原始图像路径
        seg_map:     分割结果（H x W）
        output_path: 输出图像路径
        opacity:     叠加透明度
    """
    # 读取原图
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 生成彩色分割图
    color_seg = LOVEDA_PALETTE[seg_map]
    if color_seg.shape[:2] != img_rgb.shape[:2]:
        color_seg = cv2.resize(
            color_seg, (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    overlay = (img_rgb * (1 - opacity) + color_seg * opacity).astype(np.uint8)

    # 创建图例
    legend_patches = []
    for cls_id, cls_name in enumerate(LOVEDA_CLASSES):
        color = LOVEDA_PALETTE[cls_id] / 255.0
        patch = mpatches.Patch(color=color, label=f'{cls_id}: {cls_name}')
        legend_patches.append(patch)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(color_seg)
    axes[1].set_title('语义分割结果', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('叠加效果', fontsize=14)
    axes[2].axis('off')

    # 添加图例
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=len(LOVEDA_CLASSES),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比可视化图已保存：{output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='港口图像单张推理')
    parser.add_argument(
        '--config',
        default='configs/fcn_hr48_loveda.py',
        help='模型配置文件路径')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='模型权重文件路径')
    parser.add_argument(
        '--img',
        required=True,
        help='输入图像路径')
    parser.add_argument(
        '--output',
        default='results/inference',
        help='输出目录（默认：results/inference）')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='推理设备（默认：cuda:0）')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='叠加透明度，0-1（默认：0.5）')
    return parser.parse_args()


def main():
    args = parse_args()

    from mmseg.apis import init_model

    # 初始化模型
    print(f"正在加载模型：{args.config}")
    print(f"模型权重：{args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 单张推理
    seg_map = inference_single(
        model, args.img, args.output, opacity=args.opacity)

    # 生成对比可视化图
    img_name = os.path.splitext(os.path.basename(args.img))[0]
    vis_path = os.path.join(args.output, f'{img_name}_comparison.png')
    visualize_with_legend(args.img, seg_map, vis_path, opacity=args.opacity)


if __name__ == '__main__':
    main()
