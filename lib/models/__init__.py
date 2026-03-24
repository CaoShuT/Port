#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/models/__init__.py
复用 HRNet 官方模型注册模块。
"""

import glob
import os.path as osp
import sys

# 将当前目录加入 path，支持模型模块直接导入
this_dir = osp.dirname(__file__)

from .seg_hrnet import get_seg_model     # noqa: F401
from . import seg_hrnet                   # noqa: F401
from . import seg_hrnet_ocr               # noqa: F401
