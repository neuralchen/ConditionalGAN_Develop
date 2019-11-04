#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: ConditionalWSConv2d.py
# Created Date: Tuesday October 1st 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 5th October 2019 12:21:23 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConditionalWSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, num_classes, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ConditionalWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.conditionalWeight  = nn.Embedding(num_classes, in_channels)
        self.conditionalBias    = nn.Embedding(num_classes,out_channels)
        # self.conditionalBias=nn.Linear(num_classes,out_channels*out_channels,bias=False)
        
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.num_classes=num_classes
        #print(num_classes)
        self.kernel_size=kernel_size
        self.weight_g = nn.Parameter(torch.ones(self.weight.size()[0],self.weight.size()[1],self.weight.size()[2],self.weight.size()[3])/9.0, requires_grad=False)
        # self.weight_g = nn.Parameter(torch.ones(self.weight.size()[0],self.weight.size()[1],1,1))
        # self.weight_g=torch.ones(self.weight.size()[0],self.weight.size()[1],self.weight.size()[2],self.weight.size()[3]).div(self.kernel_size[0]**2).cuda()
        self._initialize()

    def _initialize(self):
        # init.zeros_(self.conditionalBias.weight.data)
        init.uniform_(self.conditionalBias.weight.data,-1/np.sqrt(self.num_classes),-1/np.sqrt(self.num_classes))
        init.xavier_uniform_(self.weight)
        
    def forward(self, x, c):
        
        weight = self.weight
        weight_mean = weight.mean(dim=1,
                                  keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-12
        weight  = weight / std.expand_as(weight)
        #print(weight.size())
        # beta    = self.conditionalWeight(c)[0]
        if c is not None:
            
           
            X1= F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            X2= F.conv2d(x, self.weight_g, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            
            gamma   = self.conditionalBias(c)
            gamma   = gamma.view(gamma.size()[0],gamma.size()[1],1,1)
            # bs,ch,w,h= X2.size()
            # gamma   = gamma.view(bs,self.in_channels,self.in_channels)
            # # X2=torch.einsum('ijmn,ij->ijmn',X2,gamma)
            # X2  = X2.view(bs,self.out_channels,-1)
            # X2  = torch.matmul(X2.permute(0,2,1),gamma)
            # X2  = X2.permute(0,2,1).view(bs,ch,w,h)
            X2 = X2 *gamma
            
            return X1+X2