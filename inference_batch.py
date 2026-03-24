#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理脚本：对目录下所有遥感图像进行语义分割推理，
生成叠加图、彩色分割图、灰度类别索引图，并输出汇总统计表。

使用示例：
    python inference_batch.py \
        --config configs/fcn_hr48_loveda.py \
        --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
        --img-dir port_images/ \
        --output-dir results/batch/
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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

# 支持的图像格式
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def get_image_list(img_dir):
    """扫描目录，返回所有图像文件路径列表。"""
    img_dir = os.path.abspath(img_dir)
    img_files = []
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTENSIONS:
            img_files.append(os.path.join(img_dir, fname))
    return img_files


def batch_inference(model, img_dir, output_dir, opacity=0.5):
    """
    批量推理主函数。

    对每张图像输出三类结果：
    - overlay/：叠加了分割结果的原始图像
    - seg_map/：彩色分割图
    - seg_gray/：灰度类别索引图（像素值=类别索引）

    同时输出 CSV 格式的汇总统计表。

    Args:
        model:      MMSeg 推理模型
        img_dir:    输入图像目录
        output_dir: 输出根目录
        opacity:    叠加透明度（0-1）
    """
    from mmseg.apis import inference_model

    # 创建输出子目录
    overlay_dir = os.path.join(output_dir, 'overlay')
    seg_map_dir = os.path.join(output_dir, 'seg_map')
    seg_gray_dir = os.path.join(output_dir, 'seg_gray')
    for d in [overlay_dir, seg_map_dir, seg_gray_dir]:
        os.makedirs(d, exist_ok=True)

    # 获取图像列表
    img_files = get_image_list(img_dir)
    if not img_files:
        print(f"[错误] 未找到图像文件：{img_dir}")
        return

    print(f"共找到 {len(img_files)} 张图像，开始批量推理...")

    # 统计结果
    stats = []

    for img_path in tqdm(img_files, desc='推理进度'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            # 推理
            result = inference_model(model, img_path)
            seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

            # 读取原图
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # 调整分割图尺寸
            if seg_map.shape != (h, w):
                seg_map = cv2.resize(
                    seg_map.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST)

            # 生成彩色分割图
            color_seg = LOVEDA_PALETTE[seg_map]

            # 生成叠加图
            overlay = (img_rgb * (1 - opacity) +
                       color_seg * opacity).astype(np.uint8)

            # 保存结果
            Image.fromarray(overlay).save(
                os.path.join(overlay_dir, img_name + '.png'))
            Image.fromarray(color_seg).save(
                os.path.join(seg_map_dir, img_name + '.png'))
            Image.fromarray(seg_map.astype(np.uint8)).save(
                os.path.join(seg_gray_dir, img_name + '.png'))

            # 统计各类像素占比
            total = seg_map.size
            row = {'image': img_name}
            for cls_id, cls_name in enumerate(LOVEDA_CLASSES):
                count = np.sum(seg_map == cls_id)
                row[cls_name] = f"{count / total * 100:.2f}%"
            stats.append(row)

        except Exception as e:
            print(f"\n[错误] 处理 {img_path} 时出错：{e}")
            continue

    # 输出汇总统计表
    if stats:
        print("\n" + "=" * 80)
        print("批量推理汇总统计")
        print("=" * 80)

        # 表头
        header = f"{'图像名称':<30}" + "".join(
            f"{cls:<14}" for cls in LOVEDA_CLASSES)
        print(header)
        print("-" * 80)

        # 数据行
        for row in stats:
            line = f"{row['image']:<30}" + "".join(
                f"{row[cls]:<14}" for cls in LOVEDA_CLASSES)
            print(line)

        # 保存 CSV
        csv_path = os.path.join(output_dir, 'statistics.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write('image,' + ','.join(LOVEDA_CLASSES) + '\n')
            # 写入数据
            for row in stats:
                f.write(row['image'] + ',' +
                        ','.join(row[cls] for cls in LOVEDA_CLASSES) + '\n')

        print(f"\n统计表已保存至：{csv_path}")
        print(f"叠加图目录：{overlay_dir}")
        print(f"彩色分割图目录：{seg_map_dir}")
        print(f"灰度索引图目录：{seg_gray_dir}")
        print(f"总计处理：{len(stats)}/{len(img_files)} 张图像")


def parse_args():
    parser = argparse.ArgumentParser(description='批量语义分割推理')
    parser.add_argument(
        '--config',
        default='configs/fcn_hr48_loveda.py',
        help='模型配置文件路径')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='模型权重文件路径')
    parser.add_argument(
        '--img-dir',
        required=True,
        help='输入图像目录')
    parser.add_argument(
        '--output-dir',
        default='results/batch',
        help='输出目录（默认：results/batch）')
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

    # 批量推理
    batch_inference(
        model,
        args.img_dir,
        args.output_dir,
        opacity=args.opacity)


if __name__ == '__main__':
    main()
