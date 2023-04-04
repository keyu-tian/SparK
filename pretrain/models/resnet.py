# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn.functional as F
from timm.models.resnet import ResNet


# hack: inject the `get_downsample_ratio` function into `timm.models.resnet.ResNet`
def get_downsample_ratio(self: ResNet) -> int:
    return 32


# hack: inject the `get_feature_map_channels` function into `timm.models.resnet.ResNet`
def get_feature_map_channels(self: ResNet) -> List[int]:
    # `self.feature_info` is maintained by `timm`
    return [info['num_chs'] for info in self.feature_info[1:]]


# hack: override the forward function of `timm.models.resnet.ResNet`
def forward(self, x, hierarchical=False):
    """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
    >>> ResNet.forward
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    
    if hierarchical:
        ls = []
        x = self.layer1(x); ls.append(x)
        x = self.layer2(x); ls.append(x)
        x = self.layer3(x); ls.append(x)
        x = self.layer4(x); ls.append(x)
        return ls
    else:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


ResNet.get_downsample_ratio = get_downsample_ratio
ResNet.get_feature_map_channels = get_feature_map_channels
ResNet.forward = forward


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet50')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
