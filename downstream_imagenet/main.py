# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import time

import torch
import torch.distributed as tdist
from timm.utils import ModelEmaV2
from torch.utils.tensorboard import SummaryWriter

from arg import get_args, FineTuneArgs
from models import ConvNeXt, ResNet
__for_timm_registration = ConvNeXt, ResNet
from lr_decay import lr_wd_annealing
from util import init_distributed_environ, create_model_opt, load_checkpoint, save_checkpoint
from data import create_classification_dataset


def main_ft():
    world_size, global_rank, local_rank, device = init_distributed_environ()
    args: FineTuneArgs = get_args(world_size, global_rank, local_rank, device)
    print(f'initial args:\n{str(args)}')
    args.log_epoch()
    
    criterion, mixup_fn, model_without_ddp, model, model_ema, optimizer = create_model_opt(args)
    ep_start, performance_desc = load_checkpoint(args.resume_from, model_without_ddp, model_ema, optimizer)
    
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [FT already done]    Max/Last Acc: {performance_desc}')
    else:
        tb_lg = SummaryWriter(args.tb_lg_dir) if args.is_master else None
        loader_train, iters_train, iterator_val, iters_val = create_classification_dataset(
            args.data_path, args.img_size, args.rep_aug,
            args.dataloader_workers, args.batch_size_per_gpu, args.world_size, args.global_rank
        )
        
        # train & eval
        tot_pred, last_acc = evaluate(args.device, iterator_val, iters_val, model)
        max_acc = last_acc
        max_acc_e = last_acc_e = evaluate(args.device, iterator_val, iters_val, model_ema.module)[-1]
        print(f'[fine-tune] initial acc={last_acc:.2f}, ema={last_acc_e:.2f}')
        
        ep_eval = set(range(0, args.ep//3, 5)) | set(range(args.ep//3, args.ep))
        print(f'[FT start] ep_eval={sorted(ep_eval)} ')
        print(f'[FT start] from ep{ep_start}')
        
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
        ft_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            if hasattr(loader_train, 'sampler') and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(ep)
                if 0 <= ep <= 3:
                    print(f'[loader_train.sampler.set_epoch({ep})]')
            
            train_loss, train_acc = fine_tune_one_epoch(ep, args, tb_lg, loader_train, iters_train, criterion, mixup_fn, model, model_ema, optimizer, params_req_grad)
            if ep in ep_eval:
                eval_start_time = time.time()
                tot_pred, last_acc = evaluate(args.device, iterator_val, iters_val, model)
                tot_pred_e, last_acc_e = evaluate(args.device, iterator_val, iters_val, model_ema.module)
                eval_cost = round(time.time() - eval_start_time, 2)
                performance_desc = f'Max (Last) Acc: {max(max_acc, last_acc):.2f} ({last_acc:.2f} o {tot_pred})    EMA: {max(max_acc_e, last_acc_e):.2f} ({last_acc_e:.2f} o {tot_pred_e})'
                states = model_without_ddp.state_dict(), model_ema.module.state_dict(), optimizer.state_dict()
                if last_acc > max_acc:
                    max_acc = last_acc
                    save_checkpoint(f'{args.model}_1kfinetuned_best.pth', args, ep, performance_desc, *states)
                if last_acc_e > max_acc_e:
                    max_acc_e = last_acc_e
                    save_checkpoint(f'{args.model}_1kfinetuned_best_ema.pth', args, ep, performance_desc, *states)
                save_checkpoint(f'{args.model}_1kfinetuned_last.pth', args, ep, performance_desc, *states)
            else:
                eval_cost = '-'
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'[ep{ep}/{args.ep}]    {performance_desc}    Ep cost: {ep_cost}s,   Ev cost: {eval_cost},    Remain: {remain_time},    Finish @ {finish_time}')
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.train_loss, args.train_acc, args.best_val_acc = train_loss, train_acc, max(max_acc, max_acc_e)
            args.log_epoch()
            
            if args.is_master:
                tb_lg.add_scalar(f'ft_train/ep_loss', train_loss, ep)
                tb_lg.add_scalar(f'ft_eval/max_acc', max_acc, ep)
                tb_lg.add_scalar(f'ft_eval/last_acc', last_acc, ep)
                tb_lg.add_scalar(f'ft_eval/max_acc_ema', max_acc_e, ep)
                tb_lg.add_scalar(f'ft_eval/last_acc_ema', last_acc_e, ep)
                tb_lg.add_scalar(f'ft_z_burnout/rest_hours', round(remain_secs/60/60, 2), ep)
                tb_lg.flush()
        
        # finish fine-tuning
        result_acc = max(max_acc, max_acc_e)
        if args.is_master:
            tb_lg.add_scalar('ft_result/result_acc', result_acc, ep_start)
            tb_lg.add_scalar('ft_result/result_acc', result_acc, args.ep)
            tb_lg.flush()
            tb_lg.close()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [FT finished]    {performance_desc}    Total Cost: {(time.time() - ft_start_time) / 60 / 60:.1f}h\n')
        print(f'  [*] [FT finished]    max(max_acc, max_acc_e)={result_acc}    EMA better={max_acc_e>max_acc}')
        print('\n\n')
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def fine_tune_one_epoch(ep, args: FineTuneArgs, tb_lg: SummaryWriter, loader_train, iters_train, criterion, mixup_fn, model, model_ema: ModelEmaV2, optimizer, params_req_grad):
    model.train()
    tot_loss = tot_acc = 0.0
    log_freq = max(1, round(iters_train * 0.7))
    ep_start_time = time.time()
    for it, (inp, tar) in enumerate(loader_train):
        # adjust lr and wd
        cur_it = it + ep * iters_train
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, cur_it, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward
        inp = inp.to(args.device, non_blocking=True)
        raw_tar = tar = tar.to(args.device, non_blocking=True)
        if mixup_fn is not None:
            inp, tar, raw_tar = mixup_fn(inp, tar)
        oup = model(inp)
        pred = oup.data.argmax(dim=1)
        if mixup_fn is None:
            acc = pred.eq(tar).float().mean().item() * 100
            tot_acc += acc
        else:
            acc = (pred.eq(raw_tar) | pred.eq(raw_tar.flip(0))).float().mean().item() * 100
            tot_acc += acc
        
        # backward
        optimizer.zero_grad()
        loss = criterion(oup, tar)
        loss.backward()
        loss = loss.item()
        tot_loss += loss
        if args.clip > 0:
            orig_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        else:
            orig_norm = None
        optimizer.step()
        model_ema.update(model)
        torch.cuda.synchronize()
        
        # log
        if args.is_master and cur_it % log_freq == 0:
            tb_lg.add_scalar(f'ft_train/it_loss', loss, cur_it)
            tb_lg.add_scalar(f'ft_train/it_acc', acc, cur_it)
            tb_lg.add_scalar(f'ft_hp/min_lr', min_lr, cur_it), tb_lg.add_scalar(f'ft_hp/max_lr', max_lr, cur_it)
            tb_lg.add_scalar(f'ft_hp/min_wd', min_wd, cur_it), tb_lg.add_scalar(f'ft_hp/max_wd', max_wd, cur_it)
            if orig_norm is not None:
                tb_lg.add_scalar(f'ft_hp/orig_norm', orig_norm, cur_it)
        
        if it in [3, iters_train//2, iters_train-1]:
            remain_secs = (iters_train-1 - it) * (time.time() - ep_start_time) / (it + 1)
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            print(f'[ep{ep} it{it:3d}/{iters_train}]    L: {loss:.4f}    Acc: {acc:.2f}    lr: {min_lr:.1e}~{max_lr:.1e}    Remain: {remain_time}')
    
    return tot_loss / iters_train, tot_acc / iters_train


@torch.no_grad()
def evaluate(dev, iterator_val, iters_val, model):
    training = model.training
    model.train(False)
    tot_pred, tot_correct = 0., 0.
    for _ in range(iters_val):
        inp, tar = next(iterator_val)
        tot_pred += tar.shape[0]
        inp = inp.to(dev, non_blocking=True)
        tar = tar.to(dev, non_blocking=True)
        oup = model(inp)
        tot_correct += oup.argmax(dim=1).eq(tar).sum().item()
    model.train(training)
    t = torch.tensor([tot_pred, tot_correct]).to(dev)
    tdist.all_reduce(t)
    return t[0].item(), (t[1] / t[0]).item() * 100.


if __name__ == '__main__':
    main_ft()
