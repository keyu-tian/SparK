# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import drop


from models.convnext import ConvNeXt
from models.resnet import ResNet
_import_resnets_for_timm_registration = (ResNet,)


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


model_alias_to_fullname = {
    'res50': 'resnet50',
    'res101': 'resnet101',
    'res152': 'resnet152',
    'res200': 'resnet200',
    'cnxS': 'convnext_small',
    'cnxB': 'convnext_base',
    'cnxL': 'convnext_large',
}
model_fullname_to_alias = {v: k for k, v in model_alias_to_fullname.items()}


pre_train_d = { # default drop_path_rate, num of para, FLOPs, downsample_ratio, num of channel
    'resnet50': [dict(drop_path_rate=0.05), 25.6, 4.1, 32, 2048],
    'resnet101': [dict(drop_path_rate=0.08), 44.5, 7.9, 32, 2048],
    'resnet152': [dict(drop_path_rate=0.10), 60.2, 11.6, 32, 2048],
    'resnet200': [dict(drop_path_rate=0.15), 64.7, 15.1, 32, 2048],
    'convnext_small': [dict(sparse=True, drop_path_rate=0.2), 50.0, 8.7, 32, 768],
    'convnext_base': [dict(sparse=True, drop_path_rate=0.3), 89.0, 15.4, 32, 1024],
    'convnext_large': [dict(sparse=True, drop_path_rate=0.4), 198.0, 34.4, 32, 1536],
}
for v in pre_train_d.values():
    v[0]['pretrained'] = False
    v[0]['num_classes'] = 0
    v[0]['global_pool'] = ''


def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):
    from encoder import SparseEncoder
    
    kwargs, params, flops, downsample_raito, fea_dim = pre_train_d[name]
    if drop_path_rate != 0:
        kwargs['drop_path_rate'] = drop_path_rate
    print(f'[build_sparse_encoder] model kwargs={kwargs}')
    return SparseEncoder(create_model(name, **kwargs), input_size=input_size, downsample_raito=downsample_raito, encoder_fea_dim=fea_dim, sbn=sbn, verbose=verbose)
