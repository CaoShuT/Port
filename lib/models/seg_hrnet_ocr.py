#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/models/seg_hrnet_ocr.py
复用 HRNet 官方 HRNet-OCR 语义分割模型。
OCR（Object Contextual Representation）是 HRNet 的进阶版本，
通过引入上下文表示提升分割精度。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bn_helper import BatchNorm2d, BN_MOMENTUM
from .seg_hrnet import (HighResolutionNet, HighResolutionModule,
                         BasicBlock, Bottleneck, BLOCKS)

logger = logging.getLogger(__name__)


# ====================== OCR 模块 ======================

class SpatialGather_Module(nn.Module):
    """
    OCR 空间聚合模块：
    根据 soft object regions 聚合像素特征，得到 object context representation。
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # (B, HW, C)
        probs = F.softmax(self.scale * probs, dim=2)  # (B, Cls, HW)
        ocr_context = torch.matmul(probs, feats)  # (B, Cls, C)
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    """OCR 对象注意力模块（2D版本）"""

    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels

        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else None

        # query / key / value 变换
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # (B, HW, key_C)

        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)  # (B, key_C, Cls)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)  # (B, Cls, key_C)

        sim_map = torch.matmul(query, key)  # (B, HW, Cls)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)  # (B, HW, key_C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)

        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class SpatialOCR_Module(nn.Module):
    """OCR 空间上下文表示模块"""

    def __init__(self, in_channels, key_channels, out_channels, scale=1,
                 dropout=0.1, bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels, scale, bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


# ====================== HRNet-OCR 主体 ======================

class HighResolutionNetOCR(nn.Module):
    """HRNet + OCR 语义分割网络"""

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNetOCR, self).__init__()

        ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS
        num_classes = config.DATASET.NUM_CLASSES

        # 复用 HRNet backbone（除最后分割头）
        self.backbone = HighResolutionNet(config, **kwargs)

        # 获取 backbone 最终输出通道数
        last_inp_channels = sum([
            extra.STAGE4.NUM_CHANNELS[i]
            for i in range(extra.STAGE4.NUM_BRANCHES)
        ])

        # OCR 辅助分类头（用于辅助损失）
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        # OCR 模块
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=config.MODEL.OCR.DROPOUT,
        )

        # 最终分类头
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # 通过 HRNet backbone 获取多尺度特征
        h, w = x.size(2), x.size(3)

        # Stem
        x_stem = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x_stem = self.backbone.relu(self.backbone.bn2(self.backbone.conv2(x_stem)))
        x_stem = self.backbone.layer1(x_stem)

        # Stage 2
        x_list = []
        for i in range(self.backbone.stage2_cfg['NUM_BRANCHES']):
            if self.backbone.transition1[i] is not None:
                x_list.append(self.backbone.transition1[i](x_stem))
            else:
                x_list.append(x_stem)
        y_list = self.backbone.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.backbone.stage3_cfg['NUM_BRANCHES']):
            if self.backbone.transition2[i] is not None:
                if i < self.backbone.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.backbone.transition2[i](y_list[i]))
                else:
                    x_list.append(self.backbone.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.backbone.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(self.backbone.stage4_cfg['NUM_BRANCHES']):
            if self.backbone.transition3[i] is not None:
                if i < self.backbone.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.backbone.transition3[i](y_list[i]))
                else:
                    x_list.append(self.backbone.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x_stage4 = self.backbone.stage4(x_list)

        # 拼接多分支特征
        x0_h, x0_w = x_stage4[0].size(2), x_stage4[0].size(3)
        feats = torch.cat([
            x_stage4[0],
            F.interpolate(x_stage4[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_stage4[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
            F.interpolate(x_stage4[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True),
        ], 1)

        # OCR 辅助预测
        out_aux = self.aux_head(feats)

        # OCR 主预测
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)

        # 上采样到原始分辨率
        out_aux = F.interpolate(out_aux, size=(h, w), mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return [out, out_aux]

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location=lambda storage, loc: storage)
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            logger.error(f'=> 预训练文件不存在: {pretrained}')
            raise RuntimeError(f'预训练文件不存在: {pretrained}')


def get_seg_model(cfg, **kwargs):
    model = HighResolutionNetOCR(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)
    return model
