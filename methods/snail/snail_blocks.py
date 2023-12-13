"""
Reference: https://github.com/eambutu/snail-pytorch
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualConv1d(nn.Module):
    """
    A casual convolution layer. This is a 1d convolution layer where the filter
    is only allowed to see the past, not the future, inputs. This is done by
    masking the inputs to the convolution layer so that the filter cannot see
    the future inputs.

    This is used in the TCBlock.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        """
        Takes input of shape (N, in_channels, T),
        returns (N, out_channels, T - dilation)
        """
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casual_conv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casual_conv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        """
        Takes input of shape (N, in_channels, T),
        returns (N, in_channels + filters, T)
        """
        xf = self.casual_conv1(input)
        xg = self.casual_conv2(input)
        activations = F.tanh(xf) * F.sigmoid(xg)  # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):
    """
    A TCBlock is a block of DenseBlocks, where the number of DenseBlocks is
    determined by the sequence length of the input. The input is passed through
    each DenseBlock in turn, and the output of each DenseBlock is concatenated
    to the input of the next DenseBlock. The output of the final DenseBlock is
    the output of the TCBlock.
    """

    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i + 1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length, 2))))])

    def forward(self, input):
        """
        Takes input of shape (N, T, in_channels),
        returns (N, T, in_channels + filters * num_dense_blocks)
        """
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):
    """
    An AttentionBlock is a block that performs attention over the input. It
    takes the input and produces a query, key, and value vector for each
    element in the input. It then computes the attention over the input using
    these query, key, and value vectors, and concatenates the input with the
    output of the attention.
    """

    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        """
        Takes input of shape (N, T, in_channels),
        N being the batch size, T being the sequence length,
        returns (N, T, in_channels + value_size)
        """
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask).cuda()

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        query = self.linear_query(input)  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        temp = torch.bmm(temp, values)  # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2)  # shape: (N, T, in_channels + value_size)
