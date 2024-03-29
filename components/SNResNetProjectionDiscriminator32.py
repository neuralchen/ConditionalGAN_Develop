######################################################################
#  script name  : SNResNetProjectionDiscriminator32.py
#  author       : Chen Xuanhong
#  created time : 2019/9/12 01:19
#  modification time ：2019/9/12 01:19
#  modified by  : Chen Xuanhong
######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from components.DisResBlock import DisResBlock
from components.OptimizedBlock import OptimizedBlock

class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = DisResBlock(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = DisResBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = DisResBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = DisResBlock(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.block6 = DisResBlock(num_features * 16, num_features * 16,
                            activation=activation, downsample=True)
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
            
        return output