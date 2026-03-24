#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_results.py
从 TensorBoard 事件文件读取训练曲线（train_loss 和 valid_mIoU）并绘图保存。
"""

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无显示器环境


def parse_args():
    parser = argparse.ArgumentParser(description='训练曲线可视化')
    parser.add_argument('--log-dir', type=str, default='log',
                        help='TensorBoard 日志根目录')
    parser.add_argument('--output', type=str, default='output/training_curves.png',
                        help='输出图像路径')
    parser.add_argument('--smoothing', type=float, default=0.6,
                        help='曲线平滑因子 (0=不平滑, 1=最大平滑)')
    parser.add_argument('--max-epoch', type=int, default=None,
                        help='只显示前 N 个 epoch（可选）')
    return parser.parse_args()


def load_tensorboard_events(log_dir):
    """
    读取 TensorBoard 事件文件，提取 train_loss 和 valid_mIoU。
    返回: {'train_loss': [(step, value), ...], 'valid_mIoU': [(step, value), ...]}
    """
    try:
        from tensorboardX.event_file_loader import EventFileLoader
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        loader = 'tensorflow'
    except ImportError:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            loader = 'tensorflow'
        except ImportError:
            loader = 'manual'

    data = {'train_loss': [], 'valid_mIoU': []}

    # 搜索所有事件文件
    event_files = glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'),
                            recursive=True)
    if not event_files:
        event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

    if not event_files:
        print(f'[WARN] 在 {log_dir} 中未找到 TensorBoard 事件文件')
        return data

    print(f'=> 找到 {len(event_files)} 个事件文件')

    if loader == 'tensorflow':
        for event_file in event_files:
            try:
                ea = EventAccumulator(event_file)
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                for tag in tags:
                    if 'train_loss' in tag or 'loss' in tag.lower():
                        events = ea.Scalars(tag)
                        data['train_loss'].extend([(e.step, e.value) for e in events])
                    elif 'mIoU' in tag or 'miou' in tag.lower() or 'valid' in tag.lower():
                        events = ea.Scalars(tag)
                        data['valid_mIoU'].extend([(e.step, e.value) for e in events])
            except Exception as e:
                print(f'[WARN] 读取事件文件失败: {e}')
    else:
        print('[WARN] 未安装 tensorboard，尝试手动解析...')
        data = _manual_parse_events(event_files)

    # 按 step 排序
    data['train_loss'].sort(key=lambda x: x[0])
    data['valid_mIoU'].sort(key=lambda x: x[0])

    return data


def _manual_parse_events(event_files):
    """简单的手动事件文件解析（备用方案）"""
    data = {'train_loss': [], 'valid_mIoU': []}
    # 尝试用 struct 解析二进制事件文件（简化版）
    print('[INFO] 请安装 tensorboard: pip install tensorboard')
    return data


def smooth_curve(values, smoothing=0.6):
    """指数移动平均平滑"""
    if smoothing <= 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * smoothing + (1 - smoothing) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(data, output_path, smoothing=0.6, max_epoch=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('HRNet LoveDA 训练曲线', fontsize=14, fontweight='bold')

    # ---- 训练损失 ----
    ax1 = axes[0]
    if data['train_loss']:
        steps, values = zip(*data['train_loss'])
        steps = np.array(steps)
        values = np.array(values)
        if max_epoch is not None:
            mask = steps <= max_epoch
            steps, values = steps[mask], values[mask]

        ax1.plot(steps, values, alpha=0.3, color='#1f77b4', linewidth=0.8,
                 label='原始损失')
        if len(values) > 1:
            smoothed = smooth_curve(values.tolist(), smoothing)
            ax1.plot(steps, smoothed, color='#1f77b4', linewidth=2,
                     label=f'平滑损失 (α={smoothing})')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '无训练损失数据', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12)
        ax1.set_title('训练损失曲线')

    # ---- 验证 mIoU ----
    ax2 = axes[1]
    if data['valid_mIoU']:
        steps, values = zip(*data['valid_mIoU'])
        steps = np.array(steps)
        values = np.array(values)
        if max_epoch is not None:
            mask = steps <= max_epoch
            steps, values = steps[mask], values[mask]

        ax2.plot(steps, values, 'o-', color='#ff7f0e', linewidth=2,
                 markersize=4, label='mIoU')

        # 标注最佳点
        best_idx = np.argmax(values)
        ax2.annotate(f'最佳: {values[best_idx]:.4f}',
                     xy=(steps[best_idx], values[best_idx]),
                     xytext=(steps[best_idx], values[best_idx] + 0.02),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     color='red', fontsize=9)

        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('mIoU')
        ax2.set_title('验证集 mIoU 曲线')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        print(f'=> 最佳 mIoU: {values[best_idx]:.4f} (步数 {steps[best_idx]})')
    else:
        ax2.text(0.5, 0.5, '无验证 mIoU 数据', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('验证集 mIoU 曲线')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'=> 训练曲线已保存: {output_path}')


def main():
    args = parse_args()

    print(f'=> 读取 TensorBoard 日志: {args.log_dir}')
    data = load_tensorboard_events(args.log_dir)

    train_count = len(data['train_loss'])
    miou_count = len(data['valid_mIoU'])
    print(f'=> 读取到训练损失: {train_count} 条')
    print(f'=> 读取到验证mIoU: {miou_count} 条')

    if train_count == 0 and miou_count == 0:
        print('[ERROR] 未读取到任何数据，请检查日志目录')
        return

    plot_training_curves(data, args.output, args.smoothing, args.max_epoch)


if __name__ == '__main__':
    main()
