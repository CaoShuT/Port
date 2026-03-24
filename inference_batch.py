#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_batch.py
港口图像批量推理脚本
对指定目录下所有图像进行语义分割推理，保存结果并生成汇总报告。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 将 lib 目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from config import config, update_config

# 复用单张推理中的类别定义和工具函数
from inference_port import (
    CLASS_NAMES, CLASS_COLORS,
    load_model, preprocess_image,
    sliding_window_inference, multi_scale_inference,
    colorize_prediction
)


def parse_args():
    parser = argparse.ArgumentParser(description='港口图像批量语义分割推理')
    parser.add_argument('--cfg', type=str,
                        default='experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml',
                        help='YACS 配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入图像目录')
    parser.add_argument('--output-dir', type=str, default='output/batch_inference',
                        help='输出目录')
    parser.add_argument('--ext', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff'],
                        help='处理的图像扩展名')
    parser.add_argument('--sliding-window', action='store_true',
                        help='使用滑窗推理')
    parser.add_argument('--multi-scale', action='store_true',
                        help='使用多尺度推理')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='多尺度推理尺度列表')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备: cuda 或 cpu')
    parser.add_argument('--save-color', action='store_true', default=True,
                        help='保存彩色分割图')
    parser.add_argument('--save-overlay', action='store_true', default=True,
                        help='保存叠加图')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批次大小（当前版本支持 =1）')
    parser.add_argument('opts',
                        help='通过命令行覆盖配置项',
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()


def collect_images(input_dir, extensions):
    """收集目录下所有指定扩展名的图像文件"""
    input_dir = Path(input_dir)
    images = []
    for ext in extensions:
        images.extend(sorted(input_dir.glob(f'*{ext}')))
        images.extend(sorted(input_dir.glob(f'*{ext.upper()}')))
    return sorted(set(images))


def infer_single(model, image_tensor, original_shape, crop_size, device,
                 num_classes, sliding_window=False, multi_scale=False,
                 scales=None, align_corners=True):
    """对单张图像进行推理，返回预测类别图"""
    if scales is None:
        scales = [1.0]
    h, w = original_shape

    if multi_scale:
        probs = multi_scale_inference(
            model, image_tensor, scales, crop_size, device,
            num_classes, align_corners)
    elif sliding_window:
        probs = sliding_window_inference(
            model, image_tensor, crop_size, device,
            num_classes, align_corners=align_corners)
    else:
        img_resized = F.interpolate(image_tensor, size=crop_size,
                                    mode='bilinear', align_corners=align_corners)
        img_resized = img_resized.to(device)
        with torch.no_grad():
            pred = model(img_resized)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        probs = F.interpolate(pred, size=(h, w), mode='bilinear',
                              align_corners=align_corners)

    pred_map = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred_map


def generate_report(results, output_dir):
    """生成批量推理汇总报告"""
    report_path = os.path.join(output_dir, 'inference_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('港口图像批量推理汇总报告\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'处理图像总数: {len(results)}\n\n')

        # 各类别统计汇总
        f.write('各类别像素占比统计:\n')
        f.write('-' * 50 + '\n')
        f.write(f'{"类别":<15}')
        for cls_name in CLASS_NAMES:
            f.write(f'{cls_name[:8]:>10}')
        f.write('\n')

        for img_name, cls_pcts in results:
            f.write(f'{img_name[:15]:<15}')
            for pct in cls_pcts:
                f.write(f'{pct:>9.1f}%')
            f.write('\n')

        # 计算平均值
        if results:
            all_pcts = np.array([r[1] for r in results])
            mean_pcts = all_pcts.mean(axis=0)
            f.write('-' * 50 + '\n')
            f.write(f'{"平均":<15}')
            for pct in mean_pcts:
                f.write(f'{pct:>9.1f}%')
            f.write('\n')

    print(f'\n=> 汇总报告已保存: {report_path}')


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
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_color:
        os.makedirs(os.path.join(args.output_dir, 'color'), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output_dir, 'overlay'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'seg'), exist_ok=True)

    # 加载模型
    model = load_model(config, args.model, device)

    # 收集图像
    images = collect_images(args.input_dir, args.ext)
    if not images:
        print(f'[ERROR] 在 {args.input_dir} 中未找到图像文件')
        return

    print(f'=> 找到 {len(images)} 张图像，开始批量推理...\n')

    crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    num_classes = config.DATASET.NUM_CLASSES
    align_corners = config.MODEL.ALIGN_CORNERS

    results = []
    total_time = 0.0
    failed = []

    for img_path in tqdm(images, desc='推理进度'):
        img_name = img_path.stem

        try:
            # 预处理
            image_tensor, original_bgr, (h, w) = preprocess_image(str(img_path))

            # 推理
            t0 = time.time()
            pred_map = infer_single(
                model, image_tensor, (h, w), crop_size, device,
                num_classes,
                sliding_window=args.sliding_window,
                multi_scale=args.multi_scale,
                scales=args.scales,
                align_corners=align_corners
            )
            elapsed = time.time() - t0
            total_time += elapsed

            # 统计各类别占比
            cls_pcts = []
            total_pixels = h * w
            for cls_id in range(num_classes):
                count = int(np.sum(pred_map == cls_id))
                cls_pcts.append(count / total_pixels * 100)
            results.append((img_name, cls_pcts))

            # 保存分割图（类别索引）
            seg_path = os.path.join(args.output_dir, 'seg', f'{img_name}.png')
            Image.fromarray(pred_map).save(seg_path)

            # 保存彩色分割图
            if args.save_color:
                color_seg = colorize_prediction(pred_map)
                color_path = os.path.join(args.output_dir, 'color', f'{img_name}_color.png')
                Image.fromarray(color_seg).save(color_path)

            # 保存叠加图
            if args.save_overlay:
                color_seg = colorize_prediction(pred_map)
                original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
                overlay = (original_rgb.astype(np.float32) * 0.5 +
                           color_seg.astype(np.float32) * 0.5).astype(np.uint8)
                overlay_path = os.path.join(args.output_dir, 'overlay', f'{img_name}_overlay.png')
                Image.fromarray(overlay).save(overlay_path)

        except Exception as e:
            print(f'\n[WARN] 处理 {img_path.name} 失败: {e}')
            failed.append(img_path.name)
            continue

    # 汇总统计
    success_count = len(images) - len(failed)
    avg_time = total_time / max(success_count, 1)

    print(f'\n============================')
    print(f' 批量推理完成')
    print(f'============================')
    print(f' 成功处理: {success_count}/{len(images)} 张')
    print(f' 平均耗时: {avg_time:.3f} s/张')
    print(f' 总耗时:   {total_time:.1f} s')
    if failed:
        print(f' 失败图像: {failed}')
    print(f' 结果保存至: {args.output_dir}')

    # 生成汇总报告
    if results:
        generate_report(results, args.output_dir)


if __name__ == '__main__':
    main()
