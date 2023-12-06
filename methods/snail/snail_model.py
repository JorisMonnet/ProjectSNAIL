import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import math
from backbones import AttentionBlock, TCBlock

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate

class SnailModel(nn.Module):

    def __init__(self, features, n_way, n_support, num_channels):
        super(SnailModel, self).__init__(features, n_way, n_support, change_way=False)

        self.features = features  # TODO  
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))

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
        x = self.features(x)
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
        return x # TODO
