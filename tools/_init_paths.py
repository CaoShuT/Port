#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径初始化工具（复用 HRNet 官方 tools/_init_paths.py）
将 lib/ 目录加入 sys.path，使 lib 下的模块可直接导入。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# 将 lib/ 目录加入 Python 路径
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
