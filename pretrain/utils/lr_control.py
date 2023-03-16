# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from pprint import pformat


def lr_wd_annealing(optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it):
    wp_it = round(wp_it)
    if cur_it < wp_it:
        cur_lr = 0.005 * peak_lr + 0.995 * peak_lr * cur_it / wp_it
    else:
        ratio = (cur_it - wp_it) / (max_it - 1 - wp_it)
        cur_lr = 0.001 * peak_lr + 0.999 * peak_lr * (0.5 + 0.5 * math.cos(math.pi * ratio))
    
    ratio = cur_it / (max_it - 1)
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * ratio))
    
    min_lr, max_lr = cur_lr, cur_lr
    min_wd, max_wd = cur_wd, cur_wd
    for param_group in optimizer.param_groups:
        scaled_lr = param_group['lr'] = cur_lr * param_group.get('lr_scale', 1)  # 'lr_scale' could be assigned
        min_lr, max_lr = min(min_lr, scaled_lr), max(max_lr, scaled_lr)
        scaled_wd = param_group['weight_decay'] = cur_wd * param_group.get('weight_decay_scale', 1)  # 'weight_decay_scale' could be assigned
        min_wd, max_wd = min(min_wd, scaled_wd), max(max_wd, scaled_wd)
    return min_lr, max_lr, min_wd, max_wd


def get_param_groups(model, nowd_keys=()):
    para_groups, para_groups_dbg = {}, {}
    
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_ft_param_groups] param groups = \n{pformat(para_groups_dbg, indent=2, width=250)}\n')
    return list(para_groups.values())
