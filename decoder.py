# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from timm.models.layers import trunc_normal_, DropPath, Mlp
import torch.nn as nn

from utils.misc import is_pow2n

_BN = None


class UNetBlock2x(nn.Module):
    def __init__(self, cin, cout, cmid, last_act=True):
        super().__init__()
        if cmid == 0:
            c_mid = cin
        elif cmid == 1:
            c_mid = (cin + cout) // 2
            
        self.b = nn.Sequential(
            nn.Conv2d(cin, c_mid, 3, 1, 1, bias=False), _BN(c_mid), nn.ReLU6(inplace=True),
            nn.Conv2d(c_mid, cout, 3, 1, 1, bias=False), _BN(cout), (nn.ReLU6(inplace=True) if last_act else nn.Identity()),
        )
        
    def forward(self, x):
        return self.b(x)


class DecoderConv(nn.Module):
    def __init__(self, cin, cout, double, heavy, cmid):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin, kernel_size=4 if double else 2, stride=2, padding=1 if double else 0, bias=True)
        ls = [UNetBlock2x(cin, (cin if i != heavy[1]-1 else cout), cmid=cmid, last_act=i != heavy[1]-1) for i in range(heavy[1])]
        self.conv = nn.Sequential(*ls)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, decoder_fea_dim, upsample_ratio, double=False, heavy=None, cmid=0, sbn=False):
        global _BN
        _BN = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        super().__init__()
        self.fea_dim = decoder_fea_dim
        if heavy is None:
            heavy = [0, 1]
        heavy[1] = max(1, heavy[1])
        self.double_bool = double
        self.heavy = heavy
        self.cmid = cmid
        self.sbn = sbn

        assert is_pow2n(upsample_ratio)
        n = round(math.log2(upsample_ratio))
        channels = [self.fea_dim // 2**i for i in range(n+1)]
        self.dec = nn.ModuleList([
            DecoderConv(cin, cout, double, heavy, cmid) for (cin, cout) in zip(channels[:-1], channels[1:])
        ])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)
    
    def num_para(self):
        tot = sum(p.numel() for p in self.parameters())
        
        para1 = para2 = 0
        for m in self.dec.modules():
            if isinstance(m, nn.ConvTranspose2d):
                para1 += sum(p.numel() for p in m.parameters())
            elif isinstance(m, nn.Conv2d):
                para2 += sum(p.numel() for p in m.parameters())
        return f'#para: {tot/1e6:.2f} (dconv={para1/1e6:.2f}, conv={para2/1e6:.2f}, ot={(tot-para1-para2)/1e6:.2f})'

    def extra_repr(self) -> str:
        return f'fea_dim={self.fea_dim}, dbl={self.double_bool}, heavy={self.heavy}, cmid={self.cmid}, sbn={self.sbn}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight, std=.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
