# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import drop
from timm.models.resnet import ResNet

from .convnext_official import ConvNeXt


def convnext_get_layer_id_and_scale_exp(self: ConvNeXt, para_name: str):
    N = 12 if len(self.stages[-2]) > 9 else 6
    if para_name.startswith("downsample_layers"):
        stage_id = int(para_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        else:  # stage_id == 3:
            layer_id = N
    elif para_name.startswith("stages"):
        stage_id = int(para_name.split('.')[1])
        block_id = int(para_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        else:  # stage_id == 3:
            layer_id = N
    else:
        layer_id = N + 1  # after backbone
    
    return layer_id, N + 1 - layer_id


def resnets_get_layer_id_and_scale_exp(self: ResNet, para_name: str):
    # stages:
    # 50  :    [3, 4, 6, 3]
    # 101 :    [3, 4, 23, 3]
    # 152 :    [3, 8, 36, 3]
    # 200 :    [3, 24, 36, 3]
    # eca269d: [3, 30, 48, 8]
    
    L2, L3 = len(self.layer2), len(self.layer3)
    if L2 == 4 and L3 == 6:
        blk2, blk3 = 2, 3
    elif L2 == 4 and L3 == 23:
        blk2, blk3 = 2, 3
    elif L2 == 8 and L3 == 36:
        blk2, blk3 = 4, 4
    elif L2 == 24 and L3 == 36:
        blk2, blk3 = 4, 4
    elif L2 == 30 and L3 == 48:
        blk2, blk3 = 5, 6
    else:
        raise NotImplementedError
    
    N2, N3 = math.ceil(L2 / blk2 - 1e-5), math.ceil(L3 / blk3 - 1e-5)
    N = 2 + N2 + N3
    if para_name.startswith('layer'):  # 1, 2, 3, 4, 5
        stage_id, block_id = int(para_name.split('.')[0][5:]), int(para_name.split('.')[1])
        if stage_id == 1:
            layer_id = 1
        elif stage_id == 2:
            layer_id = 2 + block_id // blk2  # 2, 3
        elif stage_id == 3:
            layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5    r101: 4, 5, ..., 11
        else:  # == 4
            layer_id = N  # r50: 6       r101: 12
    elif para_name.startswith('fc.'):
        layer_id = N + 1  # r50: 7       r101: 13
    else:
        layer_id = 0
    
    return layer_id, N + 1 - layer_id  # r50: 0-7, 7-0   r101: 0-13, 13-0


def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )


# IMPORTANT: update some member functions
__UPDATED = False
if not __UPDATED:
    for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, BinaryCrossEntropy, Mixup, drop.DropPath):
        if hasattr(clz, 'extra_repr'):
            clz.extra_repr = _ex_repr
        else:
            clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'
    ResNet.get_layer_id_and_scale_exp = resnets_get_layer_id_and_scale_exp
    ConvNeXt.get_layer_id_and_scale_exp = convnext_get_layer_id_and_scale_exp
    __UPDATED = True
