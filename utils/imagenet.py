# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms

import dist

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_cls_id: int = 1000,
            only=-1,
    ):
        for postfix in (os.path.sep, 'train', 'val'):
            if root.endswith(postfix):
                root = root[:-len(postfix)]
        
        root = os.path.join(root, 'train' if train else 'val')
        
        super(ImageNetDataset, self).__init__(
            root,
            # loader=ImageLoader(train),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=target_transform, is_valid_file=is_valid_file
        )
        
        if only > 0:
            g = torch.Generator()
            g.manual_seed(0)
            idx = torch.randperm(len(self.samples), generator=g).numpy().tolist()

            ws = dist.get_world_size()
            res = (max_cls_id * only) % ws
            more = 0 if res == 0 else (ws - res)
            max_total = max_cls_id * only + more
            if (max_total // ws) % 2 == 1:
                more += ws
                max_total += ws
            
            d = {c: [] for c in range(max_cls_id)}
            max_len = {c: only for c in range(max_cls_id)}
            for c in range(max_cls_id-more, max_cls_id):
                max_len[c] += 1
            
            total = 0
            for i in idx:
                path, target = self.samples[i]
                if len(d[target]) < max_len[target]:
                    d[target].append((path, target))
                    total += 1
                if total == max_total:
                    break
            sp = []
            [sp.extend(l) for l in d.values()]

            print(f'[ds] more={more}, len(sp)={len(sp)}')
            self.samples = tuple(sp)
            self.targets = tuple([s[1] for s in self.samples])
        else:
            self.samples = tuple(filter(lambda item: item[-1] < max_cls_id, self.samples))
            self.targets = tuple([s[1] for s in self.samples])
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


def build_imagenet(mode, data_path, data_set, img_size, eval_crop_pct=None, rrc=0.3, aa='rand-m7-mstd0.5', re_prob=0.0, colorj=0.4):
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    norm = transforms.Normalize(mean=mean, std=std)

    if img_size >= 384:
        trans_val = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=interpolation),
            transforms.ToTensor(),
            norm,
        ])
    else:
        trans_val = transforms_imagenet_eval(
            img_size=img_size, interpolation='bicubic', crop_pct=eval_crop_pct,
            mean=mean, std=std
        )
    
    mode = mode.lower()
    if mode == 'pt':
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(rrc, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    elif mode == 'le':
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    else:
        trans_train = create_transform(
            is_training=True,
            input_size=img_size,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=re_prob,
            re_mode='pixel',
            re_count=1,
            color_jitter=colorj,
            mean=mean, std=std,
        )
    
    if data_path.endswith(os.path.sep):
        data_path = data_path[:-len(os.path.sep)]
    for postfix in ('train', 'val'):
        if data_path.endswith(postfix):
            data_path = data_path[:-len(postfix)]

    if data_set == 'imn':
        dataset_train = ImageNetDataset(root=data_path, transform=trans_train, train=True)
        dataset_val = ImageNetDataset(root=data_path, transform=trans_val, train=False)
        num_classes = 1000
    else:
        raise NotImplementedError

    print_transform(trans_train, '[train]')
    print_transform(trans_val, '[val]')
    
    return dataset_train, dataset_val


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
