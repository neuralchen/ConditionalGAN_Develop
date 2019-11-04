#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GenResBlockBC.py
# Created Date: Tuesday October 1st 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 6th October 2019 8:01:03 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################




import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from components.BC import BC

def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2))

class GenResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(GenResBlock, self).__init__()

        self.activation     = activation
        self.upsample       = upsample
        self.learnable_sc   = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes
        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = BC(in_ch, num_classes)
            self.b2 = BC(h_ch, num_classes)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))