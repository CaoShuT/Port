#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/utils/distributed.py
复用 HRNet 官方分布式训练工具函数。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.distributed as dist


def is_distributed():
    """判断当前是否处于分布式训练模式"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """获取当前进程在分布式训练中的 rank"""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取分布式训练的总进程数"""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor):
    """在所有进程间对 tensor 求平均"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt
