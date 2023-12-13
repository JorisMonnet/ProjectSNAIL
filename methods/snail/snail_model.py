import math

import numpy as np
import torch
import torch.nn as nn

from methods.snail.snail_blocks import AttentionBlock, TCBlock


class SnailModel(nn.Module):
    """
    Implementation of the SNAIL model for few-shot learning.
    """

    def __init__(self, features, n_way, n_support, architecture=None):
        super(SnailModel, self).__init__()

        if architecture is None:
            print("YOU FOOL YOU FORGOT TO SPECIFY THE ARCHITECTURE")

        self.n_channels = features.final_feat_dim + n_way

        self.features = features
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))

        self.snail_blocks = nn.ModuleList()

        def add_module(module, att_key_size=None, att_value_size=None, tc_filters=None, add_channels=None):
            # Usually, value size = 2 * key size and n_channels += value_size
            if module == 'attention':
                print("Adding Attention")
                attention_block = AttentionBlock(self.n_channels, att_key_size, att_value_size)
                self.snail_blocks.append(attention_block)
                if add_channels is not None:
                    self.n_channels += add_channels
                else:
                    self.n_channels += att_value_size
            # Usually, num_filters = 128 and n_channels += num_filters * 128
            elif module == 'tc':
                print("Adding TC")
                tc_block = TCBlock(self.n_channels, n_way * n_support + 1, tc_filters)
                self.snail_blocks.append(tc_block)
                if add_channels is not None:
                    self.n_channels += add_channels
                else:
                    self.n_channels += num_filters * tc_filters

        # parse architecture and use add_module
        for block in architecture:
            if block['module'] == 'attention':
                add_module(self, module='attention', att_key_size=block['att_key_size'],
                           att_value_size=block['att_value_size'])
            elif block['module'] == 'tc':
                add_module(self, module='tc', tc_filters=block['tc_filters'])
            else:
                print(f"Unrecognized module: {block['module']}")

        self.fc = nn.Linear(self.n_channels, n_way)
        self.N = n_way
        self.K = n_support

    def forward(self, x, labels):
        """
        Takes input of shape (N, T, in_channels),
        returns (N, T, in_channels + filters * num_dense_blocks)
        """
        x = self.features(x)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]

        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        x = torch.cat((x, labels), 1)

        x = x.view((batch_size, self.N * self.K + 1, -1))

        for block in self.snail_blocks:
            x = block(x)

        x = self.fc(x)

        return x
