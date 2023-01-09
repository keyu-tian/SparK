# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is basically a copy to: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from encoder import SparseConvNeXtBlock, SparseConvNeXtLayerNorm


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True,
                 ):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SparseConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first", sparse=sparse)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first", sparse=sparse),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SparseConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                      layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.depths = depths
        
        self.apply(self._init_weights)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(dims[-1], eps=1e-6, sparse=False)  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(dims[-1], num_classes)
            # self.fc.weight.data.mul_(head_init_scale)     # todo: perform this outside
            # self.fc.bias.data.mul_(head_init_scale)       # todo: perform this outside
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()
        
        self.with_pooling = len(global_pool) > 0
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x, pyramid: int): # pyramid: 0, 1, 2, 3, 4
        ls = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if pyramid:
                ls.append(x)
        
        if pyramid:
            for i in range(len(ls)-pyramid-1, -1, -1):
                del ls[i]
            return [None] * (4 - pyramid) + ls
        else:
            if self.with_pooling:
                x = x.mean([-2, -1])    # global average pooling, (N, C, H, W) -> (N, C)
            return x
    
    def forward(self, x, pyramid=0):
        if pyramid == 0:
            x = self.forward_features(x, pyramid=pyramid)
            x = self.fc(self.norm(x))
            return x
        else:
            return self.forward_features(x, pyramid=pyramid)
    
    def get_classifier(self):
        return self.fc
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        N = 12 if self.depths[-2] > 9 else 6
        if para_name.startswith("downsample_layers"):
            stage_id = int(para_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = N
        elif para_name.startswith("stages"):
            stage_id = int(para_name.split('.')[1])
            block_id = int(para_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = N
        else:
            layer_id = N + 1  # after backbone
        
        return layer_id, N + 1 - layer_id


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    from timm.models import create_model
    
    c = create_model('convnext_small', sparse=False)
    with torch.no_grad():
        x = torch.rand(2, 3, 224, 224)
        print(c(x).shape)
        print([None if f is None else f.shape for f in c(x, pyramid=1)])
        print([None if f is None else f.shape for f in c(x, pyramid=2)])
        print([None if f is None else f.shape for f in c(x, pyramid=3)])
        print([None if f is None else f.shape for f in c(x, pyramid=4)])
