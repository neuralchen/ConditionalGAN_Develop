#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: BC.py
# Created Date: Saturday September 28th 2019
# Author: Liu Naiyuan
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 25th October 2019 1:28:43 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import torch
from torch import nn
import numpy as np
from torch.nn import init

class BC(nn.Module):
    def __init__(self, in_channel, n_class=13):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(in_channel,momentum=0.1, affine=False)
        self.conditionalconv=ConditionalConv(n_class,in_channel)
        # self.unconditionalconv=UnconditionalConv(in_channel)
        


    def forward(self, x, condition):
        out = self.batch_norm(x)

        if condition is not None:
            output=self.conditionalconv(out,condition)#+self.unconditionalconv(out)
        else:
            output=self.unconditionalconv(out)
        return output

class UnconditionalConv(nn.Module):
    def __init__(self,in_channels):
        super(UnconditionalConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,in_channels,1,1)
        # nn.init.xavier_uniform(self.conv.weight)
        # nn.init.zeros_(self.conv.bias)
        
    def forward(self,x):
        output=self.conv(x)
        return output

class ConditionalConv(nn.Module):
    def __init__(self,number_of_classes,in_channels):
        super(ConditionalConv,self).__init__()
        self.in_channels=in_channels
        self.number_of_classes=number_of_classes
        # self.linearkernel=nn.Linear(number_of_classes,in_channels*in_channels,bias=False)
        # self.linearbias=nn.Linear(number_of_classes,in_channels,bias=False)
        self.linearkernel   = nn.Embedding(number_of_classes, in_channels*in_channels)
        self.linearbias     = nn.Embedding(number_of_classes, in_channels)
        init.constant_(self.linearkernel.weight.data,1/float(number_of_classes))
        init.zeros_(self.linearbias.weight.data)
        
    def forward(self,x,condition):
        bs,c,w,h=x.size()
        condition_kernel=self.linearkernel(condition)
        condition_bias=self.linearbias(condition)
        condition_kernel=condition_kernel.view(bs,self.in_channels,self.in_channels)

        x=x.view(bs,c,w*h)
        bias=torch.unsqueeze(condition_bias,dim=-1)
        bias=torch.unsqueeze(bias,dim=-1)
        output=x.permute(0,2,1).matmul(condition_kernel)
        output=output.permute(0,2,1).view(bs,c,w,h)
        output=output+bias

        return output