#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRNet 官方训练脚本（复用自 HRNet/HRNet-Semantic-Segmentation HRNet-OCR 分支）
核心流程: parse_args -> build model -> prepare data -> build criterion -> train/validate loop -> save checkpoint
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import datasets
import models
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='训练语义分割网络')
    # 必须通过 --cfg 指定配置文件
    parser.add_argument('--cfg',
                        help='YACS 配置文件路径（yaml）',
                        default='experiments/loveda/seg_hrnet_w48_train_512x512_sgd_lr1e-2_wd5e-4_bs_12_epoch200.yaml',
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help='通过命令行覆盖配置项',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


def get_sampler(dataset):
    """根据是否分布式训练选择合适的采样器"""
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # 设置随机种子
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # cuDNN 配置
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # 分布式训练初始化
    from utils.distributed import is_distributed, get_rank, get_world_size
    distributed = is_distributed()

    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    else:
        gpus = list(config.GPUS)

    # 构建模型
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    # 打印模型参数量
    if distributed:
        this_rank = get_rank()
        if this_rank == 0:
            if torch.cuda.is_available():
                dump_input = torch.rand(
                    (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                )
                logger.info(get_model_summary(model.cuda(), dump_input.cuda()))
    else:
        if torch.cuda.is_available():
            dump_input = torch.rand(
                (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
            logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # 复制配置文件到输出目录
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    # 构建 TensorBoard writer
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 准备数据集
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
    )

    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus) if not distributed else config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

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

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus) if not distributed else config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # 构建损失函数
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    model = FullModel(model, criterion)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # 构建优化器
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set(config.TRAIN.NONBACKBONE_KEYWORDS)
            for k, param in params_dict.items():
                if any(part in k for part in nbb_keys):
                    nbb_lr.append(param)
                else:
                    bb_lr.append(param)
            optimizer = torch.optim.SGD([
                {'params': bb_lr, 'lr': config.TRAIN.LR},
                {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}
            ],
                lr=config.TRAIN.LR,
                momentum=config.TRAIN.MOMENTUM,
                weight_decay=config.TRAIN.WD,
                nesterov=config.TRAIN.NESTEROV,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.TRAIN.LR,
                momentum=config.TRAIN.MOMENTUM,
                weight_decay=config.TRAIN.WD,
                nesterov=config.TRAIN.NESTEROV,
            )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAIN.LR,
        )
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WD,
        )
    else:
        raise ValueError(f'不支持的优化器: {config.TRAIN.OPTIMIZER}')

    epoch_iters = int(len(train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU /
                      (get_world_size() if distributed else len(gpus)))
    best_mIoU = 0
    last_epoch = config.TRAIN.BEGIN_EPOCH

    # 恢复 checkpoint（如果配置了 RESUME）
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint.get('best_mIoU', 0)
            last_epoch = checkpoint.get('epoch', 0)
            dct = checkpoint['state_dict']
            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in dct.items()
                 if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"=> 从 epoch {last_epoch} 恢复训练，最佳 mIoU={best_mIoU:.4f}")

    start = last_epoch
    end = config.TRAIN.END_EPOCH

    for epoch in range(start, end):
        if distributed:
            train_sampler.set_epoch(epoch)

        train(config, epoch, end, epoch_iters,
              config.TRAIN.LR, num_iters=end * epoch_iters,
              trainloader=trainloader, optimizer=optimizer,
              model=model, writer_dict=writer_dict)

        # 验证
        if (not distributed) or (distributed and get_rank() == 0):
            valid_loss, mean_IoU, IoU_array = validate(
                config, testloader, model, writer_dict)
            logger.info(f'Epoch [{epoch}/{end}] mIoU={mean_IoU:.4f}')
            logger.info(f'IoU_array: {IoU_array}')

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
                logger.info(f'=> 保存最佳模型，mIoU={best_mIoU:.4f}')

            # 保存最新 checkpoint
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

    # 保存最终模型
    if (not distributed) or (distributed and get_rank() == 0):
        torch.save(model.module.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))
        writer_dict['writer'].close()
        logger.info(f'训练完成，最佳 mIoU={best_mIoU:.4f}')
        logger.info(f'模型保存至: {final_output_dir}')


if __name__ == '__main__':
    main()
