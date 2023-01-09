# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
from timm.models.resnet import ResNet


def forward_features(self, x, pyramid: int): # pyramid: 0, 1, 2, 3, 4
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    
    ls = []
    x = self.layer1(x)
    if pyramid: ls.append(x)
    x = self.layer2(x)
    if pyramid: ls.append(x)
    x = self.layer3(x)
    if pyramid: ls.append(x)
    x = self.layer4(x)
    if pyramid: ls.append(x)
    
    if pyramid:
        for i in range(len(ls)-pyramid-1, -1, -1):
            del ls[i]
        return [None] * (4 - pyramid) + ls
    else:
        return x


def forward(self, x, pyramid=0):
    if pyramid == 0:
        x = self.forward_features(x, pyramid=pyramid)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x
    else:
        return self.forward_features(x, pyramid=pyramid)


def resnets_get_layer_id_and_scale_exp(self, para_name: str):
    # stages:
    # 50  :    [3, 4, 6, 3]
    # 101 :    [3, 4, 23, 3]
    # 152 :    [3, 8, 36, 3]
    # 200 :    [3, 24, 36, 3]
    # eca269d: [3, 30, 48, 8]
    
    L2, L3 = len(self.layer2), len(self.layer3)
    if L2 == 4 and L3 == 6:
        blk2, blk3 = 2, 3
    elif L2 == 4 and L3 == 23:
        blk2, blk3 = 2, 3
    elif L2 == 8 and L3 == 36:
        blk2, blk3 = 4, 4
    elif L2 == 24 and L3 == 36:
        blk2, blk3 = 4, 4
    elif L2 == 30 and L3 == 48:
        blk2, blk3 = 5, 6
    else:
        raise NotImplementedError
    
    N2, N3 = math.ceil(L2 / blk2 - 1e-5), math.ceil(L3 / blk3 - 1e-5)
    N = 2 + N2 + N3
    if para_name.startswith('layer'):   # 1, 2, 3, 4, 5
        stage_id, block_id = int(para_name.split('.')[0][5:]), int(para_name.split('.')[1])
        if stage_id == 1:
            layer_id = 1
        elif stage_id == 2:
            layer_id = 2 + block_id // blk2 # 2, 3
        elif stage_id == 3:
            layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5    r101: 4, 5, ..., 11
        else: # == 4
            layer_id = N                    # r50: 6       r101: 12
    elif para_name.startswith('fc.'):
        layer_id = N+1                      # r50: 7       r101: 13
    else:
        layer_id = 0
    
    return layer_id, N+1 - layer_id         # r50: 0-7, 7-0   r101: 0-13, 13-0


ResNet.get_layer_id_and_scale_exp = resnets_get_layer_id_and_scale_exp
ResNet.forward_features = forward_features
ResNet.forward = forward


if __name__ == '__main__':
    import torch
    from timm.models import create_model
    r = create_model('resnet50')
    with torch.no_grad():
        print(r(torch.rand(2, 3, 224, 224)).shape)
        print(r(torch.rand(2, 3, 224, 224), pyramid=1))
        print(r(torch.rand(2, 3, 224, 224), pyramid=2))
        print(r(torch.rand(2, 3, 224, 224), pyramid=3))
        print(r(torch.rand(2, 3, 224, 224), pyramid=4))
