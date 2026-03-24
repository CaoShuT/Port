#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/datasets/cityscapes.py
复用 HRNet 官方 Cityscapes 数据集实现，供参考。
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


class Cityscapes(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=19,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=None,
                 std=None,
                 bd_dilate_size=4):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                                        crop_size, 1, scale_factor,
                                        mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.bd_dilate_size = bd_dilate_size

        self.img_list = [line.strip().split() for line in
                         open(os.path.join(root, list_path))]
        self.files = self.read_files()

        # Cityscapes id -> trainId 映射（19 类）
        self.label_mapping = {
            -1: ignore_label, 0: ignore_label, 1: ignore_label,
            2: ignore_label, 3: ignore_label, 4: ignore_label,
            5: ignore_label, 6: ignore_label, 7: 0, 8: 1,
            9: ignore_label, 10: ignore_label, 11: 2, 12: 3,
            13: 4, 14: ignore_label, 15: ignore_label, 16: ignore_label,
            17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            29: ignore_label, 30: ignore_label, 31: 16, 32: 17,
            33: 18
        }
        self.class_weights = torch.FloatTensor([
            0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
            1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
            1.0865, 1.0955, 1.0865, 1.1529, 1.0507
        ]).cuda()

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                'img': image_path,
                'label': label_path,
                'name': name,
            })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item['name']
        image = cv2.imread(os.path.join(self.root, item['img']),
                           cv2.IMREAD_COLOR)
        size = image.shape

        label = cv2.imread(os.path.join(self.root, item['label']),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image, False)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(
            np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
