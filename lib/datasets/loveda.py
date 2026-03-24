#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/datasets/loveda.py
LoveDA 数据集类（核心新增文件）
继承 BaseDataset，遵循官方 BaseDataset / cityscapes.py 接口约定。
LoveDA 7 类别:
  0=background, 1=building, 2=road, 3=water, 4=barren, 5=forest, 6=agriculture
标注像素值直接为类别索引 (0-6)，不需要 id->trainId 映射。
ignore_label=255。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from .base_dataset import BaseDataset


class LoveDA(BaseDataset):
    """
    LoveDA 语义分割数据集。
    参数:
        root (str): 数据根目录，含 images/ labels/ train.lst val.lst
        list_path (str): lst 文件路径（相对于 root），每行格式:
                         images/{split}/{filename} labels/{split}/{filename}
        num_classes (int): 类别数，默认 7
        multi_scale (bool): 是否多尺度增强
        flip (bool): 是否水平翻转增强
        ignore_label (int): 忽略标签值，默认 255
        base_size (int): 多尺度基础尺寸
        crop_size (tuple): 裁剪尺寸 (H, W)
        scale_factor (int): 多尺度随机缩放因子
        mean (list): 归一化均值 [R, G, B]
        std (list): 归一化标准差 [R, G, B]
    """

    # LoveDA 7 类别名称
    CLASS_NAMES = [
        'background',   # 0
        'building',     # 1
        'road',         # 2
        'water',        # 3
        'barren',       # 4
        'forest',       # 5
        'agriculture',  # 6
    ]

    # 类别颜色（RGB），用于可视化
    CLASS_COLORS = np.array([
        [0,   0,   0],    # background - 黑色
        [255,  0,   0],   # building   - 红色
        [255, 255,  0],   # road       - 黄色
        [0,   0, 255],    # water      - 蓝色
        [159, 129, 183],  # barren     - 紫色
        [0,  255,   0],   # forest     - 绿色
        [255, 195, 128],  # agriculture- 橙色
    ], dtype=np.uint8)

    def __init__(self,
                 root,
                 list_path,
                 num_classes=7,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=512,
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=None,
                 std=None,
                 downsample_rate=1,
                 class_weights=None):

        super(LoveDA, self).__init__(
            ignore_label=ignore_label,
            base_size=base_size,
            crop_size=crop_size,
            downsample_rate=downsample_rate,
            scale_factor=scale_factor,
            mean=mean,
            std=std,
        )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip

        # 读取 lst 文件
        lst_full_path = os.path.join(root, list_path)
        self.img_list = [line.strip().split() for line in
                         open(lst_full_path, 'r', encoding='utf-8')
                         if line.strip()]
        self.files = self._read_files()

        # 类别权重（可选，若 None 则不加权）
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None

    def _read_files(self):
        """解析 lst 文件，返回 {'img': path, 'label': path, 'name': str} 列表"""
        files = []
        for item in self.img_list:
            if len(item) == 2:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    'img': image_path,
                    'label': label_path,
                    'name': name,
                })
            elif len(item) == 1:
                # 测试集，无标注
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    'img': image_path,
                    'label': None,
                    'name': name,
                })
        return files

    def __getitem__(self, index):
        """
        返回 (image, label, size, name)
        image: float32 tensor (3, H, W)，已归一化
        label: int32 ndarray (H, W)，像素值为类别索引 0-6 或 255(ignore)
        size: ndarray，原始图像尺寸 (H, W, C)
        name: str，图像文件名（不含扩展名）
        """
        item = self.files[index]
        name = item['name']

        # 读取图像（BGR）
        image = cv2.imread(
            os.path.join(self.root, item['img']),
            cv2.IMREAD_COLOR
        )
        if image is None:
            raise FileNotFoundError(
                f"无法读取图像: {os.path.join(self.root, item['img'])}")
        size = image.shape  # (H, W, C)

        # 读取标注
        if item['label'] is not None:
            label = cv2.imread(
                os.path.join(self.root, item['label']),
                cv2.IMREAD_GRAYSCALE
            )
            if label is None:
                raise FileNotFoundError(
                    f"无法读取标注: {os.path.join(self.root, item['label'])}")
            # LoveDA 标注：像素值直接为类别索引 0-6，无需映射
            # 将超出范围的值设为 ignore_label
            label = label.astype(np.int32)
            label[label > self.num_classes - 1] = self.ignore_label
        else:
            # 测试集无标注，创建全零 dummy label
            label = np.zeros(image.shape[:2], dtype=np.int32)

        image, label = self.gen_sample(
            image, label,
            self.multi_scale, self.flip
        )

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)

    def save_pred(self, preds, sv_path, name):
        """
        保存预测结果为 PNG 文件（像素值为类别索引）。
        preds: Tensor (N, num_classes, H, W)
        sv_path: 输出目录
        name: 文件名列表（不含扩展名）
        """
        os.makedirs(sv_path, exist_ok=True)
        pred_maps = np.asarray(
            np.argmax(preds.cpu().numpy(), axis=1), dtype=np.uint8)
        for i in range(pred_maps.shape[0]):
            save_img = Image.fromarray(pred_maps[i])
            save_img.save(os.path.join(sv_path, name[i] + '.png'))

    def get_palette(self):
        """返回 PIL Image 所需的调色板（256*3 扁平列表）"""
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[:len(self.CLASS_COLORS)] = self.CLASS_COLORS
        return palette.flatten().tolist()
