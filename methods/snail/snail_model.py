import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import math
from methods.snail.snail_blocks import AttentionBlock, TCBlock


class SnailModel(nn.Module):

    def __init__(self, features, n_way, n_support, architecture=None):
        super(SnailModel, self).__init__()

        if architecture is None:
            architecture = [
                {'module': 'attention', 'att_key_size': 64, 'att_value_size': 32},
                {'module': 'tc', 'tc_filters': 128},
                {'module': 'attention', 'att_key_size': 256, 'att_value_size': 128},
                {'module': 'tc', 'tc_filters': 128},
                {'module': 'attention', 'att_key_size': 512, 'att_value_size': 256}
            ]

        # TODO how to define num_channels ? output of FCNET
        self.n_channels = features.final_feat_dim + n_way

        self.features = features  
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))

        self.snail_blocks = nn.ModuleList()

        def add_module(self, module, att_key_size=None, att_value_size=None, tc_filters=None, add_channels=None):
            # Usually, value size = 2 * key size and n_channels += value_size
            if (module == 'attention'):
                attention_block = AttentionBlock(self.n_channels,  att_key_size, att_value_size)
                self.snail_blocks.append(attention_block)
                if add_channels:
                    self.n_channels += add_channels
                else:
                    self.n_channels += att_value_size
            # Usually, num_filters = 128 and n_channels += num_filters * 128
            elif (module == 'tc'):
                tc_block = TCBlock(self.n_channels, n_way * n_support + 1, tc_filters)
                self.snail_blocks.append(tc_block)
                if add_channels:
                    self.n_channels += add_channels
                else:
                    self.n_channels += num_filters * tc_filters

        
        # parse architecture and use add_module
        for block in architecture:
            if block['module'] == 'attention':
                add_module(self, module='attention', att_key_size=block['att_key_size'], att_value_size=block['att_value_size'])
            elif block['module'] == 'tc':
                add_module(self, module='tc', tc_filters=block['tc_filters'])
            else:
                print(f"Unrecognized module: {block['module']}")

        self.fc = nn.Linear(self.n_channels, n_way)
        self.N = n_way
        self.K = n_support

    
    def forward(self, x, labels):
        x = self.features(x)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]

        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        x = torch.cat((x, labels), 1) # TODO why do we concat labels like that?

        x = x.view((batch_size, self.N * self.K + 1, -1))

        for block in self.snail_blocks:
            x = block(x)

        x = self.fc(x)

        return x
