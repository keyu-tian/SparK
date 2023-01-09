# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys

from tap import Tap

import dist

line_sep = f'\n{"=" * 80}\n'


class Args(Tap):
    # environment
    local_rank: int         # useless
    exp_name: str
    data_path: str
    exp_dir: str
    log_txt_name: str = '/some/path/like/this/log.txt'
    resume: str = ''
    seed: int = 1
    device: str = 'cpu'
    
    # key MIM hp
    mask: float = 0.6
    mask2: float = -1
    uni: bool = False
    pe: bool = False
    pn: int = 1
    py: int = 4
    # other MIM hp
    den: bool = False
    loss_l2: bool = True
    en_de_norm: str = 'bn'
    en_de_lin: bool = True
    
    # encoder
    model: str = 'res50'
    model_alias: str = 'res50'
    input_size: int = 224
    sbn: bool = True
    # decoder
    dec_dim: int = 512      # [could be changed in `main.py`]
    double: bool = True
    hea: str = '0_1'
    cmid: int = 0
    
    # pre-training hyperparameters
    glb_batch_size: int = 0
    batch_size: int = 0     # batch size per GPU
    dp: float = 0.0
    base_lr: float = 2e-4
    lr: float = None
    wd: float = 0.04
    wde: float = 0.2
    ep: int = 1600
    wp_ep: int = 40
    clip: int = 5.
    opt: str = ''
    ada: float = 0.
    
    # data hyperparameters
    data_set: str = 'imn'
    rrc: float = 0.67
    bs: int = 4096
    num_workers: int = 8

    # would be added during runtime
    cmd: str = ''
    commit_id: str = ''
    commit_msg: str = ''
    last_loss = 1e9         # [would be changed in `main.py`]
    cur_phase: str = ''     # [would be changed in `main.py`]
    cur_ep: str = ''        # [would be changed in `main.py`]
    remain_time: str = ''   # [would be changed in `main.py`]
    finish_time: str = ''   # [would be changed in `main.py`]
    
    first_logging: bool = True
    
    @property
    def is_convnext(self):
        return 'convnext' in self.model or 'cnx' in self.model
    
    @property
    def is_resnet(self):
        return 'res' in self.model or 'res' in self.model_alias
    
    def __str__(self):
        return re.sub(r"(\[LE-FT\]:\s*)('\s+')?", r'\1', super(Args, self).__str__())

    def log_epoch(self):
        if not dist.is_local_master():
            return
        
        if self.first_logging:
            self.first_logging = False
            with open(self.log_txt_name, 'w') as fp:
                json.dump({
                    'name': self.exp_name, 'cmd': self.cmd, 'commit_id': self.commit_id,
                    'model': self.model, 'opt': self.opt,
                }, fp)
                print('', end='\n', file=fp)
        
        with open(self.log_txt_name, 'a') as fp:
            json.dump({
                'cur': self.cur_phase, 'cur_ep': self.cur_ep,
                'last_L': self.last_loss,
                'rema': self.remain_time, 'fini': self.finish_time,
            }, fp)


def init_dist_and_get_args():
    from utils import misc
    from models import model_alias_to_fullname, model_fullname_to_alias

    # initialize
    args = Args(explicit_bool=True).parse_args()
    misc.init_distributed_environ(exp_dir=args.exp_dir)
    
    # update args
    args.cmd = ' '.join(sys.argv[1:])
    args.commit_id = os.popen(f'git rev-parse HEAD').read().strip()
    args.commit_msg = os.popen(f'git log -1').read().strip().splitlines()[-1].strip()
    
    if args.model in model_alias_to_fullname.keys():
        args.model = model_alias_to_fullname[args.model]
    args.model_alias = model_fullname_to_alias[args.model]
    
    args.device = dist.get_device()
    args.batch_size = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size * dist.get_world_size()
    
    if args.is_resnet:
        args.opt = args.opt or 'lamb'
        args.ada = args.ada or 0.95

    if args.is_convnext:
        args.opt = args.opt or 'lamb'
        args.ada = args.ada or 0.999
        args.en_de_norm = 'ln'

    args.opt = args.opt.lower()
    args.lr = args.base_lr * args.glb_batch_size / 256
    args.wde = args.wde or args.wd
    
    if args.mask2 < 0:
        args.mask2 = args.mask
    args.mask, args.mask2 = min(args.mask, args.mask2), max(args.mask, args.mask2)

    if args.py <= 0:
        args.py = 1
    
    args.hea = list(map(int, args.hea.split('_')))

    args.log_txt_name = os.path.join(args.exp_dir, 'log.txt')
    
    return args
