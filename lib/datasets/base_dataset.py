#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/datasets/base_dataset.py
复用 HRNet 官方 BaseDataset 实现。
提供数据增强、多尺度推理等通用功能。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import cv2
import numpy as np
import torch
from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 512),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=None,
                 std=None):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.scale_factor = scale_factor
        self.downsample_rate = 1. / downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    # 图像归一化预处理
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]    # BGR -> RGB
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    # 填充图像到指定大小
    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)
        return pad_image

    # 随机裁剪
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    # 多尺度增强
    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    # 缩放到指定短边长度
    def resize_short_length(self, image, label=None, short_length=None,
                            fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = int(h * short_length / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=(0., 0., 0.))
            if label is not None:
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w,
                                           cv2.BORDER_CONSTANT, value=(self.ignore_label,))
            if return_padding:
                return image, label, (pad_h, pad_w)

        if label is not None:
            return image, label
        return image

    # 随机亮度增强
    def random_brightness(self, img):
        if not self.brightness:
            return img
        self.shift_value = getattr(self, 'shift_value', 10)
        if random.random() < 0.5:
            shift = random.randint(-self.shift_value, self.shift_value)
            image = img.astype(np.float32)
            image[:, :, :] += shift
            image = np.clip(image, 0, 255).astype(np.uint8)
            return image
        return img

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label

    def reduce_zero_label(self, labelmap):
        """将 0 转为 ignore_label，其余标签 -1"""
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        encoded_labelmap[labelmap == 0] = self.ignore_label
        return encoded_labelmap

    def inference(self, config, model, image, flip):
        """单尺度推理"""
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = torch.nn.functional.interpolate(
            input=pred,
            size=size[-2:],
            mode='bilinear',
            align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = torch.nn.functional.interpolate(
                input=flip_output,
                size=size[-2:],
                mode='bilinear',
                align_corners=config.MODEL.ALIGN_CORNERS
            )
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5

        return pred.exp()

    def multi_scale_inference(self, config, model, image,
                              scales=None, flip=False, stride_rate=2.0/3.0):
        """多尺度推理"""
        if scales is None:
            scales = [1]
        crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        h, w = image.size()[-2:]
        full_probs = torch.zeros((1, config.DATASET.NUM_CLASSES, h, w)).cuda()

        for scale in scales:
            new_h = int(scale * h)
            new_w = int(scale * w)
            image_scale = torch.nn.functional.interpolate(
                image, size=(new_h, new_w),
                mode='bilinear',
                align_corners=config.MODEL.ALIGN_CORNERS
            )
            scaled_probs = self.process_image(config, model, image_scale,
                                              crop_size, stride_rate, flip)
            probs = torch.nn.functional.interpolate(
                scaled_probs, size=(h, w),
                mode='bilinear',
                align_corners=config.MODEL.ALIGN_CORNERS
            )
            full_probs += probs

        full_probs /= len(scales)
        return full_probs

    def process_image(self, config, model, image, crop_size=None,
                      stride_rate=2.0/3.0, flip=False):
        """滑窗推理"""
        if crop_size is None:
            crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

        h, w = image.size()[-2:]
        crop_h, crop_w = crop_size

        if h <= crop_h and w <= crop_w:
            return self.inference(config, model, image, flip)

        # 滑窗
        stride_h = int(crop_h * stride_rate)
        stride_w = int(crop_w * stride_rate)
        full_probs = torch.zeros((1, config.DATASET.NUM_CLASSES, h, w)).cuda()
        count_mat = torch.zeros((1, 1, h, w)).cuda()

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
                crop_img = image[:, :, y1:y2, x1:x2]
                pad_h = max(crop_h - crop_img.shape[2], 0)
                pad_w = max(crop_w - crop_img.shape[3], 0)
                crop_img = torch.nn.functional.pad(
                    crop_img, (0, pad_w, 0, pad_h))
                pred = self.inference(config, model, crop_img, flip)
                full_probs[:, :, y1:y2, x1:x2] += pred[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1

        full_probs = full_probs / count_mat
        return full_probs
