#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本：将 LoveDA 原始数据集转换为 MMSeg 格式，
同时支持准备港口自定义数据集。
"""

import os
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# LoveDA 7 类定义
LOVEDA_CLASSES = (
    'background', 'building', 'road', 'water',
    'barren', 'forest', 'agriculture'
)

# LoveDA 调色板（RGB）
LOVEDA_PALETTE = [
    [0, 0, 0],        # background - 黑色
    [255, 0, 0],      # building - 红色
    [255, 255, 0],    # road - 黄色
    [0, 0, 255],      # water - 蓝色
    [159, 129, 183],  # barren - 紫色
    [0, 255, 0],      # forest - 绿色
    [255, 195, 128],  # agriculture - 橙色
]


def organize_loveda(src_dir, dst_dir):
    """
    将 LoveDA 原始目录结构转换为 MMSeg 格式。

    原始结构：
        Train/Val/Test -> Urban/Rural -> images_png/masks_png

    目标结构：
        img_dir/train(val/test)  - 图像文件（加 Urban_/Rural_ 前缀）
        ann_dir/train(val)       - 标注文件（加 Urban_/Rural_ 前缀）

    Args:
        src_dir: LoveDA 原始数据集根目录
        dst_dir: MMSeg 格式输出目录
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # 定义分割映射：原始目录名 -> MMSeg 目录名
    split_map = {
        'Train': 'train',
        'Val': 'val',
        'Test': 'test',
    }

    for src_split, dst_split in split_map.items():
        split_src = src_dir / src_split
        if not split_src.exists():
            print(f"[跳过] 目录不存在：{split_src}")
            continue

        img_dst = dst_dir / 'img_dir' / dst_split
        ann_dst = dst_dir / 'ann_dir' / dst_split
        img_dst.mkdir(parents=True, exist_ok=True)
        ann_dst.mkdir(parents=True, exist_ok=True)

        # 遍历 Urban 和 Rural
        for scene in ['Urban', 'Rural']:
            scene_dir = split_src / scene
            if not scene_dir.exists():
                continue

            prefix = f"{scene}_"

            # 复制图像
            images_dir = scene_dir / 'images_png'
            if images_dir.exists():
                for img_file in sorted(images_dir.glob('*.png')):
                    dst_file = img_dst / (prefix + img_file.name)
                    shutil.copy2(img_file, dst_file)
                print(f"[完成] {src_split}/{scene} 图像 -> {img_dst}")

            # 复制标注（Test 集通常没有标注）
            masks_dir = scene_dir / 'masks_png'
            if masks_dir.exists() and dst_split != 'test':
                for mask_file in sorted(masks_dir.glob('*.png')):
                    dst_file = ann_dst / (prefix + mask_file.name)
                    shutil.copy2(mask_file, dst_file)
                print(f"[完成] {src_split}/{scene} 标注 -> {ann_dst}")

    print(f"\nLoveDA 数据集已转换至：{dst_dir}")


def verify_dataset(data_root):
    """
    验证数据集完整性。

    检查内容：
    - 图像和标注文件是否匹配
    - 标注文件的像素值范围是否正确

    Args:
        data_root: MMSeg 格式的数据集根目录
    """
    data_root = Path(data_root)

    for split in ['train', 'val']:
        img_dir = data_root / 'img_dir' / split
        ann_dir = data_root / 'ann_dir' / split

        if not img_dir.exists():
            print(f"[跳过] 图像目录不存在：{img_dir}")
            continue

        img_files = sorted(img_dir.glob('*.png'))
        ann_files = sorted(ann_dir.glob('*.png')) if ann_dir.exists() else []

        print(f"\n[{split}] 图像数量：{len(img_files)}，标注数量：{len(ann_files)}")

        # 检查文件名匹配
        img_names = {f.name for f in img_files}
        ann_names = {f.name for f in ann_files}

        missing_ann = img_names - ann_names
        if missing_ann:
            print(f"  [警告] 以下图像缺少标注：{list(missing_ann)[:5]}")

        extra_ann = ann_names - img_names
        if extra_ann:
            print(f"  [警告] 以下标注没有对应图像：{list(extra_ann)[:5]}")

        # 检查标注值范围
        if ann_files:
            sample = ann_files[:min(5, len(ann_files))]
            all_values = set()
            for ann_file in sample:
                mask = np.array(Image.open(ann_file))
                all_values.update(np.unique(mask).tolist())
            print(f"  标注像素值范围（采样 {len(sample)} 张）：{sorted(all_values)}")
            print(f"  期望范围：0-{len(LOVEDA_CLASSES) - 1}（LoveDA 7 类）")

    print("\n数据集验证完成！")


def prepare_port_dataset(port_img_dir, port_mask_dir, output_dir,
                         train_ratio=0.8, random_seed=42):
    """
    准备港口自定义数据集用于微调。

    将港口图像和标注按 80/20 比例划分为训练集和验证集，
    并整理为 MMSeg 格式。

    Args:
        port_img_dir:  港口图像目录
        port_mask_dir: 港口标注目录
        output_dir:    输出目录
        train_ratio:   训练集比例，默认 0.8
        random_seed:   随机种子，默认 42
    """
    port_img_dir = Path(port_img_dir)
    port_mask_dir = Path(port_mask_dir)
    output_dir = Path(output_dir)

    # 获取所有图像文件
    img_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    img_files = []
    for ext in img_exts:
        img_files.extend(port_img_dir.glob(f'*{ext}'))
    img_files = sorted(img_files)

    if not img_files:
        print(f"[错误] 未找到图像文件：{port_img_dir}")
        return

    # 随机打乱并划分
    random.seed(random_seed)
    random.shuffle(img_files)
    n_train = int(len(img_files) * train_ratio)
    train_files = img_files[:n_train]
    val_files = img_files[n_train:]

    print(f"港口数据集划分：训练 {len(train_files)} 张，验证 {len(val_files)} 张")

    # 创建输出目录
    for split, files in [('train', train_files), ('val', val_files)]:
        img_out = output_dir / 'img_dir' / split
        ann_out = output_dir / 'ann_dir' / split
        img_out.mkdir(parents=True, exist_ok=True)
        ann_out.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            # 复制图像
            shutil.copy2(img_file, img_out / img_file.name)

            # 查找对应标注（支持不同后缀）
            mask_found = False
            for ext in img_exts:
                mask_file = port_mask_dir / (img_file.stem + ext)
                if mask_file.exists():
                    # 标注统一保存为 PNG
                    mask = Image.open(mask_file)
                    mask.save(ann_out / (img_file.stem + '.png'))
                    mask_found = True
                    break

            if not mask_found:
                print(f"  [警告] 未找到标注：{img_file.name}")

    print(f"港口数据集已准备至：{output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='数据集准备工具')
    parser.add_argument(
        '--src', type=str, default=None,
        help='LoveDA 原始数据集根目录')
    parser.add_argument(
        '--dst', type=str, default='data/loveDA',
        help='MMSeg 格式输出目录（默认：data/loveDA）')
    parser.add_argument(
        '--verify', type=str, default=None,
        help='验证指定目录下的数据集完整性')
    parser.add_argument(
        '--port-img', type=str, default=None,
        help='港口图像目录')
    parser.add_argument(
        '--port-mask', type=str, default=None,
        help='港口标注目录')
    parser.add_argument(
        '--port-output', type=str, default='data/port',
        help='港口数据集输出目录（默认：data/port）')
    return parser.parse_args()


def main():
    args = parse_args()

    # 转换 LoveDA 数据集
    if args.src is not None:
        print(f"正在转换 LoveDA 数据集：{args.src} -> {args.dst}")
        organize_loveda(args.src, args.dst)

    # 验证数据集
    if args.verify is not None:
        print(f"正在验证数据集：{args.verify}")
        verify_dataset(args.verify)

    # 准备港口数据集
    if args.port_img is not None and args.port_mask is not None:
        print(f"正在准备港口数据集：{args.port_img} -> {args.port_output}")
        prepare_port_dataset(args.port_img, args.port_mask, args.port_output)

    if args.src is None and args.verify is None and args.port_img is None:
        print("用法示例：")
        print("  # 转换 LoveDA 数据集")
        print("  python prepare_loveda.py --src /path/to/LoveDA --dst data/loveDA")
        print()
        print("  # 验证数据集")
        print("  python prepare_loveda.py --verify data/loveDA")
        print()
        print("  # 准备港口数据集")
        print("  python prepare_loveda.py --port-img /path/to/port/images "
              "--port-mask /path/to/port/masks --port-output data/port")


if __name__ == '__main__':
    main()
