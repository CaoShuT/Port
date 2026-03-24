#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练入口脚本：基于 MMSegmentation 框架训练 HRNet 语义分割模型。

使用示例：
    # 单卡训练
    python train.py --config configs/fcn_hr48_loveda.py

    # 多卡分布式训练
    python train.py --config configs/fcn_hr48_loveda.py --launcher pytorch

    # 开启混合精度训练
    python train.py --config configs/fcn_hr48_loveda.py --amp
"""

import argparse
import os

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation 训练脚本')
    parser.add_argument(
        '--config',
        default='configs/fcn_hr48_loveda.py',
        help='训练配置文件路径')
    parser.add_argument(
        '--work-dir',
        help='模型保存和日志输出目录（覆盖配置文件中的设置）')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='从最新 checkpoint 恢复训练')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='开启自动混合精度（AMP）训练')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        default=False,
        help='根据 GPU 数量自动缩放学习率')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式训练启动器')
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    # 设置启动器
    cfg.launcher = args.launcher

    # 设置 work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 根据配置文件名自动生成 work_dir
        cfg.work_dir = os.path.join(
            'work_dirs',
            os.path.splitext(os.path.basename(args.config))[0])

    # 开启混合精度训练
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('[警告] AMP 已在配置文件中启用，忽略 --amp 参数')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` 仅支持 OptimWrapper，当前为 '
                f'{optim_wrapper}')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 恢复训练
    if args.resume:
        cfg.resume = True

    # 自动缩放学习率
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                '配置文件中未找到 `auto_scale_lr` 配置项，'
                '请在配置文件中添加 `auto_scale_lr=dict(enable=True, base_batch_size=...)`')

    # 打印训练信息
    print(f"配置文件：{args.config}")
    print(f"输出目录：{cfg.work_dir}")
    print(f"混合精度：{args.amp}")
    print(f"分布式启动：{args.launcher}")

    # 构建 Runner 并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
