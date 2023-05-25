# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import sys
from functools import partial
from typing import List, Tuple, Callable

import pytz
import torch
import torch.distributed as tdist
import torch.multiprocessing as tmp
from timm import create_model
from timm.loss import SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.optim import AdamW, Lamb
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

from arg import FineTuneArgs
from downstream_imagenet.mixup import BatchMixup
from lr_decay import get_param_groups


def time_str(for_dirname=False):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%m-%d_%H-%M-%S' if for_dirname else '[%m-%d %H:%M:%S]')


def init_distributed_environ():
    # ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py#L29
    if tmp.get_start_method(allow_none=True) is None:
        tmp.set_start_method('spawn')
    global_rank, num_gpus = int(os.environ.get('RANK', 'error')), torch.cuda.device_count()
    local_rank = global_rank % num_gpus
    torch.cuda.set_device(local_rank)
    
    tdist.init_process_group(backend='nccl')
    assert tdist.is_initialized(), 'torch.distributed is not initialized!'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # print only when local_rank == 0 or print(..., force=True)
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def prt(msg, *args, **kwargs):
        force = kwargs.pop('force', False)
        if local_rank == 0 or force:
            f_back = sys._getframe().f_back
            file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
            builtin_print(f'{time_str()} ({file_desc}, line{f_back.f_lineno:-4d})=> {msg}', *args, **kwargs)
    
    __builtin__.print = prt
    tdist.barrier()
    return tdist.get_world_size(), global_rank, local_rank, torch.empty(1).cuda().device


def create_model_opt(args: FineTuneArgs) -> Tuple[torch.nn.Module, Callable, torch.nn.Module, DistributedDataParallel, ModelEmaV2, Optimizer]:
    num_classes = 1000
    model_without_ddp: torch.nn.Module = create_model(args.model, num_classes=num_classes, drop_path_rate=args.drop_path).to(args.device)
    model_para = f'{sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1e6:.1f}M'
    # create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    model_ema = ModelEmaV2(model_without_ddp, decay=args.ema, device=args.device)
    if args.sbn:
        model_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    print(f'[model={args.model}] [#para={model_para}, drop_path={args.drop_path}, ema={args.ema}] {model_without_ddp}\n')
    model = DistributedDataParallel(model_without_ddp, device_ids=[args.local_rank], find_unused_parameters=False, broadcast_buffers=False)
    model.train()
    opt_cls = {
        'adam': AdamW, 'adamw': AdamW,
        'lamb': partial(Lamb, max_grad_norm=1e7, always_adapt=True, bias_correction=False),
    }
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'}, lr_scale=args.lr_scale)
    # param_groups[0] is like this: {'params': List[nn.Parameters], 'lr': float, 'lr_scale': float, 'weight_decay': float, 'weight_decay_scale': float}
    optimizer = opt_cls[args.opt](param_groups, lr=args.lr, weight_decay=0)
    print(f'[optimizer={type(optimizer)}]')
    mixup_fn = BatchMixup(
        mixup_alpha=args.mixup, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=num_classes
    )
    mixup_fn.mixup_enabled = args.mixup > 0.0
    if 'lamb' in args.opt:
        # label smoothing is solved in AdaptiveMixup with `label_smoothing`, so here smoothing=0
        criterion = BinaryCrossEntropy(smoothing=0, target_threshold=None)
    else:
        criterion = SoftTargetCrossEntropy()
    print(f'[loss_fn] {criterion}')
    print(f'[mixup_fn] {mixup_fn}')
    return criterion, mixup_fn, model_without_ddp, model, model_ema, optimizer


def load_checkpoint(resume_from, model_without_ddp, ema_module, optimizer):
    if len(resume_from) == 0 or not os.path.exists(resume_from):
        raise AttributeError(f'ckpt `{resume_from}` not found!')
        # return 0, '[no performance_desc]'
    print(f'[try to resume from file `{resume_from}`]')
    checkpoint = torch.load(resume_from, map_location='cpu')
    assert checkpoint.get('is_pretrain', False) == False, 'Please do not use `*_withdecoder_1kpretrained_spark_style.pth`, which is ONLY for resuming the pretraining. Use `*_1kpretrained_timm_style.pth` or `*_1kfinetuned*.pth` instead.'
    
    ep_start, performance_desc = checkpoint.get('epoch', -1) + 1, checkpoint.get('performance_desc', '[no performance_desc]')
    missing, unexpected = model_without_ddp.load_state_dict(checkpoint.get('module', checkpoint), strict=False)
    print(f'[load_checkpoint] missing_keys={missing}')
    print(f'[load_checkpoint] unexpected_keys={unexpected}')
    print(f'[load_checkpoint] ep_start={ep_start}, performance_desc={performance_desc}')
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'ema' in checkpoint:
        ema_module.load_state_dict(checkpoint['ema'])
    return ep_start, performance_desc


def save_checkpoint(save_to, args, epoch, performance_desc, model_without_ddp_state, ema_state, optimizer_state):
    checkpoint_path = os.path.join(args.exp_dir, save_to)
    if args.is_local_master:
        to_save = {
            'args': str(args),
            'arch': args.model,
            'epoch': epoch,
            'performance_desc': performance_desc,
            'module': model_without_ddp_state,
            'ema': ema_state,
            'optimizer': optimizer_state,
            'is_pretrain': False,
        }
        torch.save(to_save, checkpoint_path)
