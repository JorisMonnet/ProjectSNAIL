import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import math
from backbones import AttentionBlock, TCBlock

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class Snail(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, num_channels):
        super(Snail, self).__init__(backbone, n_way, n_support, change_way=False)

        self.backbone = backbone  # TODO      
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))

        # TODO check paper implem
        self.attention1 = AttentionBlock(num_channels,  64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, n_way * n_support + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, n_way * n_support + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, n_way)
        self.N = n_way
        self.K = n_support

    
    def forward(self, x, labels):
        # TODO check paper schema implem
        x = self.backbone(input)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]

        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        x = torch.cat((x, labels), 1)

        x = x.view((batch_size, self.N * self.K + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x


    def set_forward(self, x, y=None):
        ...

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parent function
        ...

    def set_forward_loss(self, x):
        ...

    def train_loop(self, epoch, train_loader, optimizer):
        ...

    def test_loop(self, val_loader, return_std=None):
        ...

    def set_forward(self, x, y=None):
        ...

    
