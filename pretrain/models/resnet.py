# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from timm.models.resnet import ResNet


def forward(self, x, hierarchy=0):  # hierarchy: 0 or 1 or 2 or 3 or 4
    """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
    >>> ResNet.forward
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    
    ls = []
    x = self.layer1(x)
    ls.append(x if hierarchy >= 4 else None)
    x = self.layer2(x)
    ls.append(x if hierarchy >= 3 else None)
    x = self.layer3(x)
    ls.append(x if hierarchy >= 2 else None)
    x = self.layer4(x)
    ls.append(x if hierarchy >= 1 else None)
    
    if hierarchy:
        return ls
    else:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


ResNet.forward = forward


if __name__ == '__main__':
    from timm.models import create_model
    r50 = create_model('resnet50')
    
    def prt(lst):
        print([tuple(t.shape) if t is not None else '(None)' for t in lst])
    with torch.no_grad():
        inp = torch.rand(2, 3, 224, 224)
        prt(r50(inp))
        prt(r50(inp, hierarchy=1))
        prt(r50(inp, hierarchy=2))
        prt(r50(inp, hierarchy=3))
        prt(r50(inp, hierarchy=4))
