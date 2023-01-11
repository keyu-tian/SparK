# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator

import numpy as np
import pytz
import torch
import torch.distributed as tdist

import dist

os_system = functools.partial(subprocess.call, shell=True)
os_system_get_stdout = lambda cmd: subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


def time_str():
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('[%m-%d %H:%M:%S]')


def init_distributed_environ(exp_dir):
    dist.initialize()
    dist.barrier()

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    _set_print_only_on_master_proc(is_master=dist.is_local_master())
    if dist.is_local_master() and len(exp_dir):
        sys.stdout, sys.stderr = _SyncPrintToFile(exp_dir, stdout=True), _SyncPrintToFile(exp_dir, stdout=False)


def save_checkpoint(fname, args, epoch, performance_desc, model_without_ddp_state, optimizer_state):
    checkpoint_path = os.path.join(args.exp_dir, fname)
    if dist.is_local_master():
        to_save = {
            'args': str(args),
            'arch': args.model,
            'epoch': epoch,
            'performance_desc': performance_desc,
            'module': model_without_ddp_state,
            'optimizer': optimizer_state,
        }
        torch.save(to_save, checkpoint_path)
    dist.barrier()


def load_checkpoint(fname, model_without_ddp, optimizer):
    print(f'[try to resume from file `{fname}`]')
    checkpoint = torch.load(fname, map_location='cpu')
    
    next_ep, performance_desc = checkpoint['epoch'] + 1, checkpoint['performance_desc']
    missing, unexpected = model_without_ddp.load_state_dict(checkpoint['module'], strict=False)
    print(f'[load_checkpoint] missing_keys={missing}')
    print(f'[load_checkpoint] unexpected_keys={unexpected}')
    print(f'[load_checkpoint] next_ep={next_ep}, performance_desc={performance_desc}')
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return next_ep, performance_desc


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]

    def time_preds(self, counts):
        remain_secs = counts * self.median
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
        return remain_secs, str(remain_time), finish_time
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, max_iters, itrt, print_freq, header=None):
        print_iters = set(np.linspace(0, max_iters-1, print_freq, dtype=int).tolist())
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(max_iters))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for i in range(max_iters):
                obj = next(itrt)
                self.data_time.update(time.time() - end)
                yield obj
                self.iter_time.update(time.time() - end)
                if i in print_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)))
                end = time.time()
        else:
            for i, obj in enumerate(itrt):
                self.data_time.update(time.time() - end)
                yield obj
                self.iter_time.update(time.time() - end)
                if i in print_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)))
                end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}   Total time:      {}   ({:.3f} s / it)'.format(
            header, total_time_str, total_time / max_iters))


def _set_print_only_on_master_proc(is_master):
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    
    def prt(msg, *args, **kwargs):
        force = kwargs.pop('force', False)
        clean = kwargs.pop('clean', False)
        deeper = kwargs.pop('deeper', False)
        if is_master or force:
            if not clean:
                f_back = sys._getframe().f_back
                if deeper and f_back.f_back is not None:
                    f_back = f_back.f_back
                file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
                msg = f'{time_str()} ({file_desc}, line{f_back.f_lineno:-4d})=> {msg}'
            builtin_print(msg, *args, **kwargs)
    
    __builtin__.print = prt


class _SyncPrintToFile(object):
    def __init__(self, exp_dir, stdout=True):
        self.terminal = sys.stdout if stdout else sys.stderr
        fname = os.path.join(exp_dir, 'stdout_backup.txt' if stdout else 'stderr_backup.txt')
        self.log = open(fname, 'w')
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
