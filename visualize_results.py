#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练结果可视化工具：解析 MMSegmentation 训练日志，绘制 loss 曲线和 mIoU 曲线。

使用示例：
    python visualize_results.py \
        --log work_dirs/fcn_hr48_loveda/20231001_120000/vis_data/scalars.json \
        --output results/training_curves.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """
    解析 MMSegmentation JSON 格式日志文件。

    支持格式：每行一个 JSON 对象（MMEngine 日志格式）

    Args:
        log_path: JSON 日志文件路径

    Returns:
        train_iters: 训练迭代次数列表
        train_losses: 训练 loss 列表
        val_iters: 验证迭代次数列表
        val_miou: 验证 mIoU 列表
    """
    train_iters = []
    train_losses = []
    val_iters = []
    val_miou = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 训练记录：包含 loss 字段
            if 'loss' in record:
                iter_num = record.get('iter', record.get('step', None))
                if iter_num is not None:
                    train_iters.append(int(iter_num))
                    train_losses.append(float(record['loss']))

            # 验证记录：包含 mIoU 字段
            if 'mIoU' in record:
                iter_num = record.get('iter', record.get('step', None))
                if iter_num is not None:
                    val_iters.append(int(iter_num))
                    val_miou.append(float(record['mIoU']))

    return train_iters, train_losses, val_iters, val_miou


def smooth_curve(values, weight=0.6):
    """
    指数移动平均平滑曲线。

    Args:
        values: 原始数值列表
        weight: 平滑权重（0-1，越大越平滑）

    Returns:
        smoothed: 平滑后的数值列表
    """
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + val * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(train_iters, train_losses, val_iters, val_miou,
                         output_path):
    """
    绘制训练曲线：左子图为 loss 曲线（原始+平滑），右子图为 mIoU 曲线（标注最佳点）。

    Args:
        train_iters:  训练迭代次数列表
        train_losses: 训练 loss 列表
        val_iters:    验证迭代次数列表
        val_miou:     验证 mIoU 列表
        output_path:  输出图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('训练过程监控', fontsize=16, fontweight='bold')

    # -------- 左图：Loss 曲线 --------
    ax1 = axes[0]
    if train_iters and train_losses:
        # 原始 loss（半透明）
        ax1.plot(train_iters, train_losses,
                 color='#1f77b4', alpha=0.3, linewidth=0.8, label='原始 Loss')
        # 平滑 loss
        smoothed = smooth_curve(train_losses, weight=0.6)
        ax1.plot(train_iters, smoothed,
                 color='#1f77b4', linewidth=2.0, label='平滑 Loss')

        # 标注最终 loss
        final_loss = smoothed[-1]
        ax1.annotate(
            f'最终: {final_loss:.4f}',
            xy=(train_iters[-1], final_loss),
            xytext=(-60, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=10, color='#1f77b4')

    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练 Loss 曲线', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # -------- 右图：mIoU 曲线 --------
    ax2 = axes[1]
    if val_iters and val_miou:
        ax2.plot(val_iters, val_miou,
                 color='#2ca02c', linewidth=2.0, marker='o',
                 markersize=5, label='验证 mIoU')

        # 标注最佳 mIoU
        best_idx = int(np.argmax(val_miou))
        best_iter = val_iters[best_idx]
        best_miou = val_miou[best_idx]

        ax2.scatter([best_iter], [best_miou],
                    color='red', s=100, zorder=5, label=f'最佳: {best_miou:.4f}')
        ax2.annotate(
            f'最佳: {best_miou:.4f}\n(iter {best_iter})',
            xy=(best_iter, best_miou),
            xytext=(20, -30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('mIoU', fontsize=12)
    ax2.set_title('验证 mIoU 曲线', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存：{output_path}")
    if val_miou:
        best_idx = int(np.argmax(val_miou))
        print(f"最佳 mIoU：{val_miou[best_idx]:.4f}（iter {val_iters[best_idx]}）")


def parse_args():
    parser = argparse.ArgumentParser(description='训练结果可视化工具')
    parser.add_argument(
        '--log',
        required=True,
        help='MMSeg JSON 日志文件路径（通常在 work_dir/时间戳/vis_data/scalars.json）')
    parser.add_argument(
        '--output',
        default='results/training_curves.png',
        help='输出图像路径（默认：results/training_curves.png）')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"正在解析日志文件：{args.log}")
    train_iters, train_losses, val_iters, val_miou = parse_log(args.log)

    print(f"训练记录：{len(train_iters)} 条，验证记录：{len(val_miou)} 条")

    if not train_iters and not val_iters:
        print("[警告] 未解析到有效数据，请检查日志文件格式")
        return

    plot_training_curves(
        train_iters, train_losses, val_iters, val_miou, args.output)


if __name__ == '__main__':
    main()
