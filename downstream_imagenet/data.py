# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import PIL.Image as PImage
import numpy as np
import torch
import torchvision
from timm.data import AutoAugment as TimmAutoAugment
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.distributed_sampler import RepeatAugSampler
from timm.data.transforms_factory import transforms_imagenet_eval
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import AutoAugment as TorchAutoAugment
from torchvision.transforms import transforms, TrivialAugmentWide

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def create_classification_dataset(data_path, img_size, rep_aug, workers, batch_size_per_gpu, world_size, global_rank):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    trans_train = create_transform(
        is_training=True, input_size=img_size,
        auto_augment='v0', interpolation='bicubic', re_prob=0.25, re_mode='pixel', re_count=1,
        mean=mean, std=std,
    )
    if img_size < 384:
        for i, t in enumerate(trans_train.transforms):
            if isinstance(t, (TorchAutoAugment, TimmAutoAugment)):
                trans_train.transforms[i] = TrivialAugmentWide(interpolation=interpolation)
                break
        trans_val = transforms_imagenet_eval(img_size=img_size, interpolation='bicubic', crop_pct=0.95, mean=mean, std=std)
    else:
        trans_val = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=interpolation),
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
        ])
    print_transform(trans_train, '[train]')
    print_transform(trans_val, '[val]')
    
    imagenet_folder = os.path.abspath(data_path)
    for postfix in ('train', 'val'):
        if imagenet_folder.endswith(postfix):
            imagenet_folder = imagenet_folder[:-len(postfix)]
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(imagenet_folder, 'train'), trans_train)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(imagenet_folder, 'val'), trans_val)
    
    if rep_aug:
        print(f'[dataset] using repeated augmentation: count={rep_aug}')
        train_sp = RepeatAugSampler(dataset_train, shuffle=True, num_repeats=rep_aug)
    else:
        train_sp = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True, drop_last=True)
    
    loader_train = DataLoader(
        dataset=dataset_train, num_workers=workers, pin_memory=True,
        batch_size=batch_size_per_gpu, sampler=train_sp, persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn,
    )
    iters_train = len(loader_train)
    print(f'[dataset: train] bs={world_size}x{batch_size_per_gpu}={world_size * batch_size_per_gpu}, num_iters={iters_train}')
    
    val_ratio = 2
    loader_val = DataLoader(
        dataset=dataset_val, num_workers=workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(world_size, global_rank, len(dataset_val), glb_batch_size=val_ratio * batch_size_per_gpu, filling=False, shuffle=False),
        worker_init_fn=worker_init_fn,
    )
    iters_val = len(loader_val)
    print(f'[dataset: val] bs={world_size}x{val_ratio * batch_size_per_gpu}={val_ratio * world_size * batch_size_per_gpu}, num_iters={iters_val}')
    
    time.sleep(3)
    warnings.resetwarnings()
    return loader_train, iters_train, iter(loader_val), iters_val


def worker_init_fn(worker_id):
    # see: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')


class DistInfiniteBatchSampler(Sampler):
    def __init__(self, world_size, global_rank, dataset_len, glb_batch_size, seed=0, filling=False, shuffle=True):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, global_rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size
        
        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())
        
        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep
