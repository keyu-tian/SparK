#!/usr/bin/python3

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl

import torch


# we use `timm.models.ResNet` in pre-training, so keys are timm-style
def timm_resnet_to_detectron2_resnet(source_file, target_file):
    pretrained: dict = torch.load(source_file, map_location='cpu')
    for mod_k in {'state_dict', 'state', 'module', 'model'}:
        if mod_k in pretrained:
            pretrained = pretrained[mod_k]
    if any(k.startswith('module.encoder_q.') for k in pretrained.keys()):
        pretrained = {k.replace('module.encoder_q.', ''): v for k, v in pretrained.items() if k.startswith('module.encoder_q.')}
    
    pkl_state = {}
    for k, v in pretrained.items(): # convert resnet's keys from timm-style to d2-style
        if 'layer' not in k:
            k = 'stem.' + k
        for t in [1, 2, 3, 4]:
            k = k.replace(f'layer{t}', f'res{t+1}')
        for t in [1, 2, 3]:
            k = k.replace(f'bn{t}', f'conv{t}.norm')
        k = k.replace('downsample.0', 'shortcut')
        k = k.replace('downsample.1', 'shortcut.norm')
        
        pkl_state[k] = v.detach().numpy()
    
    with open(target_file, 'wb') as fp:
        print(f'[convert] .pkl is generated! (from `{source_file}`, to `{target_file}`, len(state)=={len(pkl_state)})')
        pkl.dump({'model': pkl_state, '__author__': 'https://github.com/keyu-tian/SparK', 'matching_heuristics': True}, fp)


if __name__ == '__main__':
    import sys
    timm_resnet_to_detectron2_resnet(sys.argv[1], sys.argv[2])
