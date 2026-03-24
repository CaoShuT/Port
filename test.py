#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试评估脚本：基于 MMSegmentation 框架对训练好的模型进行测试评估。

使用示例：
    # 基础测试
    python test.py --config configs/fcn_hr48_loveda.py \
        --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth

    # 保存可视化结果
    python test.py --config configs/fcn_hr48_loveda.py \
        --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
        --show-dir work_dirs/vis_results

    # TTA（测试时增强）
    python test.py --config configs/fcn_hr48_loveda.py \
        --checkpoint work_dirs/fcn_hr48_loveda/iter_80000.pth \
        --tta
"""

import argparse
import os

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation 测试脚本')
    parser.add_argument(
        '--config',
        default='configs/fcn_hr48_loveda.py',
        help='测试配置文件路径')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='模型权重文件路径')
    parser.add_argument(
        '--work-dir',
        help='保存测试结果的目录')
    parser.add_argument(
        '--show',
        action='store_true',
        help='是否实时显示预测结果')
    parser.add_argument(
        '--show-dir',
        help='保存可视化结果的目录')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='是否使用测试时增强（TTA）')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式测试启动器')
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # 开启可视化
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = 2
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            '配置文件中未找到 `visualization` hook，'
            '请在 default_hooks 中添加 SegVisualizationHook')
    return cfg


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
        cfg.work_dir = os.path.join(
            'work_dirs',
            os.path.splitext(os.path.basename(args.config))[0])

    # 加载 checkpoint
    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    # 可视化设置
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # TTA 设置
    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # 打印测试信息
    print(f"配置文件：{args.config}")
    print(f"模型权重：{args.checkpoint}")
    if args.show_dir:
        print(f"可视化输出：{args.show_dir}")
    print(f"TTA：{args.tta}")

    # 构建 Runner 并开始测试
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
