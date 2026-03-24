#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/utils/utils.py
复用 HRNet 官方工具函数。
包含: create_logger, AverageMeter, get_confusion_matrix, adjust_learning_rate, FullModel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn


def create_logger(cfg, cfg_name, phase='train'):
    """
    创建日志记录器和输出目录。
    返回: (logger, final_output_dir, tensorboard_log_dir)
    """
    root_output_dir = cfg.OUTPUT_DIR
    # 从 cfg 路径提取模型名和数据集名
    cfg_name_clean = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = os.path.join(
        root_output_dir,
        cfg.DATASET.DATASET,
        cfg.MODEL.NAME,
        cfg_name_clean
    )
    print(f'=> creating {final_output_dir}')
    os.makedirs(final_output_dir, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{cfg_name_clean}_{time_str}_{phase}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # TensorBoard 日志目录
    tensorboard_log_dir = os.path.join(
        cfg.LOG_DIR,
        cfg.DATASET.DATASET,
        cfg.MODEL.NAME,
        f'{cfg_name_clean}_{time_str}'
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    return logger, final_output_dir, tensorboard_log_dir


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    计算混淆矩阵。
    label: (B, H, W) LongTensor
    pred:  (B, C, H, W) FloatTensor（logits）
    返回: (num_class, num_class) ndarray
    """
    output = pred.data.cpu().numpy()
    seg_pred = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index, minlength=num_class ** 2)
    confusion_matrix = label_count.reshape(num_class, num_class)

    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    """
    Poly LR 学习率调度。
    lr = base_lr * (1 - cur_iters / max_iters) ^ power
    """
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr


class FullModel(nn.Module):
    """
    将模型和损失函数封装在一起，支持 DataParallel 时并行计算 loss。
    forward 返回 (loss, pred)，其中 loss 已在各 GPU 上计算完毕。
    """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        loss = self.loss(outputs, labels)
        return torch.unsqueeze(loss, 0), outputs
