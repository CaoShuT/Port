#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/core/function.py
复用 HRNet 官方训练/验证/测试核心函数。
包含: train(), validate(), testval(), test()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import AverageMeter, get_confusion_matrix, adjust_learning_rate

logger = logging.getLogger(__name__)


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    """
    训练一个 epoch。
    使用 PolyLR 学习率调度（每 iter 更新）。
    """
    # 训练模式
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PolyLR：每 iter 更新学习率
        adjust_learning_rate(optimizer,
                             base_lr,
                             num_iters,
                             i_iter + cur_iters)

        # 记录
        batch_time.update(time.time() - tic)
        tic = time.time()
        ave_loss.update(loss.item())

        if i_iter % config.PRINT_FREQ == 0:
            msg = (
                f'Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], '
                f'Time: {batch_time.average():.2f}, '
                f'lr: {[x["lr"] for x in optimizer.param_groups]}, '
                f'Loss: {ave_loss.average():.6f}'
            )
            logger.info(msg)

        writer.add_scalar('train_loss', ave_loss.average(), global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
        global_steps += 1


def validate(config, testloader, model, writer_dict=None):
    """
    验证集评估，计算 mIoU。
    返回: (mean_loss, mean_IoU, IoU_array)
    """
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            loss = losses.mean()
            ave_loss.update(loss.item())

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            pred = pred[config.TEST.OUTPUT_INDEX] if config.MODEL.NUM_OUTPUTS > 1 else pred[0]

            # 上采样到 label 分辨率
            pred = F.interpolate(
                input=pred, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL
            )

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    if writer_dict is not None:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    """
    完整测试集评估（含 pixel_acc, mean_acc, mIoU）。
    返回: (mean_IoU, IoU_array, pixel_acc, mean_acc)
    """
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, label, size, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image.cuda(),
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST
            )

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear',
                    align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label.long().cuda(),
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL
            )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                os.makedirs(sv_path, exist_ok=True)
                test_dataset.save_pred(pred, sv_path, name)

            if idx % 100 == 0:
                logger.info(f'processing: {idx} images')
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logger.info(f'mIoU: {mean_IoU:.4f}')

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model, sv_dir='', sv_pred=True):
    """
    纯推理（无标注），保存预测结果。
    """
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, size, name = batch
            size = label.size()

            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image.cuda(),
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST
            )

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear',
                    align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                os.makedirs(sv_path, exist_ok=True)
                test_dataset.save_pred(pred, sv_path, name)
