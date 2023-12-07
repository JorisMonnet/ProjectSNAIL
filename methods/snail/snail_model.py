import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import math
from methods.snail.snail_blocks import AttentionBlock, TCBlock


class SnailModel(nn.Module):

    def __init__(self, features, n_way, n_support):
        super(SnailModel, self).__init__()

        # TODO how to define num_channels ? output of FCNET
        n_channels = features.final_feat_dim + n_way

        self.features = features  
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))

        self.attention1 = AttentionBlock(n_channels,  64, 32)
        n_channels += 32
        self.tc1 = TCBlock(n_channels, n_way * n_support + 1, 128)
        n_channels += num_filters * 128
        self.attention2 = AttentionBlock(n_channels, 256, 128)
        n_channels += 128
        self.tc2 = TCBlock(n_channels, n_way * n_support + 1, 128)
        n_channels += num_filters * 128
        self.attention3 = AttentionBlock(n_channels, 512, 256)
        n_channels += 256
        self.fc = nn.Linear(n_channels, n_way)
        self.N = n_way
        self.K = n_support

    
    def forward(self, x, labels):
        x = self.features(x)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]

        print("====================")
        print("SnailModel.forward")
        print("x.size():", x.size())
        print("labels.size():", labels.size())
        print("batch_size:", batch_size)
        print("last_idxs:", last_idxs)
        print("====================")

        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        x = torch.cat((x, labels), 1)

        print("====================")
        print("SnailModel.forward after cat")
        print("x.size():", x.size())
        print("====================")

        x = x.view((batch_size, self.N * self.K + 1, -1))

        print("====================")
        print("SnailModel.forward after view")
        print("x.size():", x.size())
        print("====================")

        # TODO first attention layer doens't receive the expected size
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x # TODO
