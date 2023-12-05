from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        ('bn', nn.BatchNorm2d(out_channels, momentum=1)),
        #('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True,eps=1e-5, momentum=0.1):
    # momentum = 1 restricts stats to the current mini-batch
    # This hack only works when momentum is 1 and avoids needing to track
    # running stats by substituting dummy variables
    size = int(np.prod(np.array(input.data.size()[1])))
    running_mean = torch.zeros(size).cuda()
    running_var = torch.ones(size).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)