#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRNet 官方测试脚本（复用自 HRNet/HRNet-Semantic-Segmentation HRNet-OCR 分支）
功能：加载 config 和 model，使用 testval() 或 test() 进行评估/推理。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import datasets
import models
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='测试语义分割网络')
    parser.add_argument('--cfg',
                        help='YACS 配置文件路径（yaml）',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help='通过命令行覆盖配置项',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cuDNN 配置
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # 构建模型
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    if torch.cuda.is_available():
        dump_input = torch.rand(
            (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')

    logger.info(f'=> 加载模型: {model_state_file}')
    pretrained_dict = torch.load(model_state_file,
                                 map_location=lambda storage, loc: storage)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # 准备测试集
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # 根据是否有标注选择评估模式
    if 'test' in config.DATASET.TEST_SET.lower():
        # 纯推理，保存预测图
        test(config, test_dataset, testloader, model,
             sv_dir=final_output_dir, sv_pred=True)
    else:
        # 有标注，计算 mIoU
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(
            config, test_dataset, testloader, model, sv_dir=final_output_dir)
        msg = (f'MeanIU: {mean_IoU:.4f}, PixelAcc: {pixel_acc:.4f}, '
               f'MeanAcc: {mean_acc:.4f}, Class IoU:')
        logger.info(msg)
        logger.info(IoU_array)


if __name__ == '__main__':
    main()
