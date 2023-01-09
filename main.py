# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import meta, misc, optim
from utils.imagenet import build_imagenet
from utils.lr_control import lr_wd_annealing, get_param_groups


def main_pt():
    args: meta.Args = meta.init_dist_and_get_args()
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.log_epoch()

    # build data
    print(f'[build data for pre-training] ...\n')
    dataset_train, _ = build_imagenet('pt', args.data_path, args.data_set, args.input_size, eval_crop_pct=None, rrc=args.rrc)
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.num_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, seed=args.seed,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    itrt_train = iter(data_loader_train)
    iters_train = len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}')

    # build models (encoder, decoder, and other components)
    enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
    dec = LightDecoder(args.dec_dim, enc.downsample_raito, double=args.double, heavy=args.hea, cmid=args.cmid, sbn=args.sbn)
    spark = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask, mask_ratio2=args.mask2, uniform=args.uni,
        using_pe=args.pe, pix_norm=args.pn, dense_loss=args.den, loss_l2=args.loss_l2,
        en_de_norm=args.en_de_norm, en_de_lin=args.en_de_lin, sbn=args.sbn, pyramid=args.py,
    )
    print(f'[PT model] model = {spark}\n')
    spark.to(args.device)
    model: DistributedDataParallel = DistributedDataParallel(spark, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    model_without_ddp: SparK = model.module

    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'pos_embed', 'mask_token', 'gamma'}, lr_scale=0)
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(optim.TimmLAMB, betas=(0.9, args.ada), max_grad_norm=args.clip),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume
    next_ep, performance_desc = misc.load_checkpoint(args.resume, model_without_ddp, optimizer) if len(args.resume) else (0, '[no performance_desc]')
    if next_ep >= args.ep:
        # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Recon Loss: {performance_desc}')
    else:
        # perform pre-training
        start_time = time.time()
        min_loss = 1e9
        print(f'[PT start] from ep{next_ep}')
        
        for ep in range(next_ep, args.ep):
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats, (sec, remain_time, finish_time) = pre_train_one_ep(ep, args, itrt_train, iters_train, model, optimizer)
            last_loss = stats['last_loss']
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            print(f'  [*] [ep{ep}]    Min/Last Recon Loss: {performance_desc},    Remain: {remain_time},    Finish: {finish_time}')
            
            args.cur_phase = 'PT'
            args.cur_ep = f'{ep+1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            misc.save_checkpoint(f'ckpt-last.pth', args, ep, performance_desc, model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
    
        # finish pre-training
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - start_time) / 60 / 60:.1f}h')
        print('\n\n')

    misc.save_checkpoint(f'ckpt-final.pth', args, args.ep-1, performance_desc, model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch: [{ep:3d}/{args.ep}]'

    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = args.clip > 0 and hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    # for every batch do:
    for it, (inp, _) in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        g_it = it + ep*iters_train
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, g_it, args.wp_ep*iters_train, args.ep*iters_train)
        
        # forward and backward
        inp = inp.to(args.device, non_blocking=True)
        SparK.forward
        active_ex, rec, loss = model(inp)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)

        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip)
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()

        # log
        me.update(last_loss=loss)
        me.update(max_lr=max_lr)
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
        
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds((args.ep-1-ep) * (iters_train+10))


if __name__ == '__main__':
    main_pt()
