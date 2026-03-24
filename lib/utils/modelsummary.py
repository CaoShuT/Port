#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/utils/modelsummary.py
复用 HRNet 官方模型摘要工具（打印参数量和计算量）。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from collections import namedtuple

import torch
import torch.nn as nn


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    打印模型各层的参数量统计。
    返回包含参数摘要的字符串。
    """
    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            if class_name not in layer_instances:
                layer_instances[class_name] = 1
            else:
                layer_instances[class_name] += 1

            instance_index = layer_instances[class_name]
            layer_name = class_name + "_" + str(instance_index)

            params = 0
            if hasattr(module, 'weight'):
                params += module.weight.data.nelement()
            if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                params += module.bias.data.nelement()

            if isinstance(input[0], list):
                input_sizes = [list(i.size()) for i in input[0]]
            else:
                input_sizes = list(input[0].size())

            if isinstance(output, (list, tuple)):
                output_sizes = [list(o.size()) for o in output]
            else:
                output_sizes = list(output.size())

            summary.append(ModuleDetails(
                name=layer_name,
                input_size=input_sizes,
                output_size=output_sizes,
                num_parameters=params,
                multiply_adds=0
            ))

        if not isinstance(module, nn.ModuleList) and \
                not isinstance(module, nn.Sequential) and \
                module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    try:
        model(*input_tensors)
    except Exception as e:
        pass

    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = (
            f"Model Summary" + os.linesep +
            f"Name{' ' * (space_len - 4)}|{' ' * (space_len)}|"
            f"{'#Params':^{space_len}}|" + os.linesep +
            "-" * (space_len * 3 + 3) + os.linesep
        )

    params_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if verbose:
            details += (
                f"{layer.name[:space_len - 1]:{space_len}}|"
                f"{str(layer.input_size):{space_len}}|"
                f"{layer.num_parameters:{space_len}}|" + os.linesep
            )

    details += f'\nTotal Parameters: {params_sum:,}\n'
    return details
