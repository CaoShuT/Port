#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoveDA 数据集准备脚本
功能：
  1. 将 LoveDA 原始目录组织为 HRNet 所需目录结构
  2. 生成 train.lst / val.lst 文件
  3. 验证数据集完整性
LoveDA 7 类别: 0=background, 1=building, 2=road, 3=water, 4=barren, 5=forest, 6=agriculture
ignore_label=255（标注中像素值直接为类别索引 0-6）
"""

import os
import shutil
import argparse
from pathlib import Path


def organize_loveda(src_dir: str, dst_dir: str):
    """
    将 LoveDA 原始目录组织为 HRNet 需要的目录结构。
    原始结构（以 LoveDA 官方为准）：
        src_dir/
        ├── Train/
        │   ├── Urban/
        │   │   ├── images_png/   (或 images/)
        │   │   └── masks_png/    (或 masks/)
        │   └── Rural/
        │       ├── images_png/
        │       └── masks_png/
        └── Val/
            ├── Urban/
            └── Rural/
    目标结构：
        dst_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            └── val/
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # 创建目标目录
    for split in ['train', 'val', 'test']:
        (dst_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        (dst_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    split_map = {
        'Train': 'train',
        'Val': 'val',
        'Test': 'test',
    }

    # 遍历原始目录
    for src_split, dst_split in split_map.items():
        split_path = src_dir / src_split
        if not split_path.exists():
            print(f"[WARN] 目录不存在，跳过: {split_path}")
            continue

        # 支持 Urban/Rural 子目录，也支持直接放在 split 下
        sub_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        if not sub_dirs:
            sub_dirs = [split_path]

        img_count = 0
        lbl_count = 0
        for sub in sub_dirs:
            # 查找 images 目录
            img_dir = None
            for candidate in ['images_png', 'images', 'image']:
                if (sub / candidate).exists():
                    img_dir = sub / candidate
                    break
            if img_dir is None and sub == split_path:
                img_dir = sub

            # 查找 masks 目录（仅 train/val 有标注）
            lbl_dir = None
            if dst_split in ['train', 'val']:
                for candidate in ['masks_png', 'masks', 'mask', 'labels', 'annotations']:
                    if (sub / candidate).exists():
                        lbl_dir = sub / candidate
                        break

            # 复制图像
            if img_dir and img_dir.exists():
                for img_file in sorted(img_dir.glob('*.png')):
                    dst_file = dst_dir / 'images' / dst_split / img_file.name
                    if not dst_file.exists():
                        shutil.copy2(str(img_file), str(dst_file))
                    img_count += 1
                for img_file in sorted(img_dir.glob('*.jpg')):
                    dst_file = dst_dir / 'images' / dst_split / img_file.name
                    if not dst_file.exists():
                        shutil.copy2(str(img_file), str(dst_file))
                    img_count += 1
                for img_file in sorted(img_dir.glob('*.tif')):
                    dst_file = dst_dir / 'images' / dst_split / img_file.name
                    if not dst_file.exists():
                        shutil.copy2(str(img_file), str(dst_file))
                    img_count += 1

            # 复制标注（仅 train/val）
            if lbl_dir and lbl_dir.exists():
                for lbl_file in sorted(lbl_dir.glob('*.png')):
                    dst_file = dst_dir / 'labels' / dst_split / lbl_file.name
                    if not dst_file.exists():
                        shutil.copy2(str(lbl_file), str(dst_file))
                    lbl_count += 1

        print(f"[INFO] {dst_split}: 图像 {img_count} 张, 标注 {lbl_count} 张")

    print(f"[INFO] 数据整理完成，目标目录: {dst_dir}")


def generate_lst(data_root: str, split: str, output_lst: str):
    """
    生成 HRNet 格式的 lst 文件。
    每行格式: images/{split}/{filename} labels/{split}/{filename}
    注意：图像文件名与标注文件名对应（通常相同或相差后缀）。
    """
    data_root = Path(data_root)
    img_dir = data_root / 'images' / split
    lbl_dir = data_root / 'labels' / split

    if not img_dir.exists():
        print(f"[ERROR] 图像目录不存在: {img_dir}")
        return 0

    # 收集图像文件
    img_files = sorted(img_dir.glob('*.png'))
    img_files += sorted(img_dir.glob('*.jpg'))
    img_files += sorted(img_dir.glob('*.tif'))

    if not img_files:
        print(f"[WARN] {img_dir} 中没有找到图像文件")
        return 0

    lines = []
    missing_labels = 0
    for img_file in img_files:
        # 构造标注文件路径（优先同名 .png）
        lbl_file = lbl_dir / img_file.name
        if not lbl_file.exists():
            # 尝试 .png 后缀替换
            lbl_file = lbl_dir / (img_file.stem + '.png')

        if split in ['train', 'val']:
            if not lbl_file.exists():
                print(f"[WARN] 标注文件不存在: {lbl_file}")
                missing_labels += 1
                continue
            img_rel = f"images/{split}/{img_file.name}"
            lbl_rel = f"labels/{split}/{lbl_file.name}"
            lines.append(f"{img_rel} {lbl_rel}")
        else:
            # test split 无标注
            img_rel = f"images/{split}/{img_file.name}"
            lines.append(img_rel)

    # 写入 lst 文件
    output_lst = Path(output_lst)
    output_lst.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_lst), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"[INFO] 生成 {output_lst}: {len(lines)} 条记录 (缺少标注: {missing_labels})")
    return len(lines)


def verify_dataset(data_root: str):
    """
    验证数据集完整性：检查图像/标注数量是否一致，标注像素值范围是否正确。
    """
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] 未安装 Pillow，跳过像素值验证")
        return

    data_root = Path(data_root)
    print("\n============================")
    print(" 数据集完整性验证")
    print("============================")

    for split in ['train', 'val']:
        img_dir = data_root / 'images' / split
        lbl_dir = data_root / 'labels' / split
        lst_file = data_root / f'{split}.lst'

        if not img_dir.exists():
            print(f"[WARN] {split} 图像目录不存在")
            continue

        img_count = len(list(img_dir.glob('*.png'))) + len(list(img_dir.glob('*.jpg')))
        lbl_count = len(list(lbl_dir.glob('*.png'))) if lbl_dir.exists() else 0
        lst_count = 0
        if lst_file.exists():
            with open(str(lst_file)) as f:
                lst_count = sum(1 for line in f if line.strip())

        print(f"\n[{split.upper()}]")
        print(f"  图像数量: {img_count}")
        print(f"  标注数量: {lbl_count}")
        print(f"  LST 记录: {lst_count}")

        if img_count != lbl_count:
            print(f"  [WARN] 图像与标注数量不一致！")

        # 抽样检查像素值范围（最多 5 张）
        lbl_files = list(lbl_dir.glob('*.png'))[:5] if lbl_dir.exists() else []
        if lbl_files:
            print(f"  像素值检查（前 {len(lbl_files)} 张标注）:")
            for lf in lbl_files:
                try:
                    lbl = np.array(Image.open(str(lf)))
                    unique_vals = np.unique(lbl)
                    valid = all(0 <= int(v) < 7 or int(v) == 255 for v in unique_vals)
                    status = "✓" if valid else "✗"
                    print(f"    {status} {lf.name}: 像素值 {unique_vals.tolist()}")
                except Exception as e:
                    print(f"    [ERROR] 读取 {lf.name} 失败: {e}")

    print("\n[INFO] 验证完成")


def parse_args():
    parser = argparse.ArgumentParser(description='LoveDA 数据集准备工具')
    parser.add_argument('--src', type=str, default=None,
                        help='LoveDA 原始数据目录（含 Train/Val 子目录）')
    parser.add_argument('--dst', type=str, default='data/loveda',
                        help='目标数据目录（默认: data/loveda）')
    parser.add_argument('--skip-organize', action='store_true',
                        help='跳过数据整理步骤（已整理好时使用）')
    parser.add_argument('--verify', action='store_true',
                        help='验证数据集完整性')
    return parser.parse_args()


def main():
    args = parse_args()

    dst_dir = args.dst

    # Step 1: 整理目录（可选）
    if not args.skip_organize:
        if args.src is None:
            print("[ERROR] 请通过 --src 指定 LoveDA 原始数据目录")
            print("  示例: python prepare_loveda.py --src /path/to/LoveDA --dst data/loveda")
            return
        print(f"\n[Step 1] 整理数据目录: {args.src} -> {dst_dir}")
        organize_loveda(args.src, dst_dir)
    else:
        print(f"[Step 1] 跳过数据整理，使用目录: {dst_dir}")

    # Step 2: 生成 lst 文件
    print("\n[Step 2] 生成 lst 文件")
    train_count = generate_lst(dst_dir, 'train', os.path.join(dst_dir, 'train.lst'))
    val_count = generate_lst(dst_dir, 'val', os.path.join(dst_dir, 'val.lst'))

    print(f"\n[INFO] 生成结果: train={train_count}, val={val_count}")

    # Step 3: 验证数据集（可选）
    if args.verify:
        verify_dataset(dst_dir)

    print("\n============================")
    print(" 数据准备完成！")
    print(f" 数据目录: {dst_dir}")
    print(f" 训练集 LST: {dst_dir}/train.lst")
    print(f" 验证集 LST: {dst_dir}/val.lst")
    print("============================")


if __name__ == '__main__':
    main()
