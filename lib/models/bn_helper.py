#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/models/bn_helper.py
复用 HRNet 官方 BatchNorm 辅助模块。
支持 SyncBN / ABN / InPlaceABN 等多种 BN 实现的统一接口。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


# 默认使用 PyTorch 内置 BatchNorm2d
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1


def get_syncbn():
    """返回 SyncBatchNorm（分布式训练使用）"""
    return nn.SyncBatchNorm


class ABN(nn.Module):
    """Activated Batch Normalization（BN + 激活函数一体化）"""
    def __init__(self, num_features, activation=nn.ReLU(inplace=True)):
        super(ABN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum=BN_MOMENTUM)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(x))
