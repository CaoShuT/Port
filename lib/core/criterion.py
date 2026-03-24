#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/core/criterion.py
复用 HRNet 官方损失函数（CrossEntropy + OhemCrossEntropy）。
支持多输出（BALANCE_WEIGHTS）。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    """标准交叉熵损失（支持 ignore_label 和类别权重）"""

    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w),
                mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if isinstance(score, (list, tuple)):
            # 多输出时取平均
            return sum(self._forward(s, target) for s in score) / len(score)
        return self._forward(score, target)


class OhemCrossEntropy(nn.Module):
    """
    OHEM（Online Hard Example Mining）交叉熵损失。
    只计算预测概率低于 thres 的困难像素的损失。
    """

    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w),
                mode='bilinear', align_corners=True)

        return self.criterion(score, target)

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w),
                mode='bilinear', align_corners=True)

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        if isinstance(score, (list, tuple)):
            # 多输出：第一个用 OHEM，其余用标准 CE
            functions = [self._ohem_forward] + \
                        [self._ce_forward for _ in range(len(score) - 1)]
            return sum(fn(s, target) for fn, s in zip(functions, score))
        return self._ohem_forward(score, target)
