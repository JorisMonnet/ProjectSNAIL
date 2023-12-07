import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate
from methods.snail.snail_model import SnailModel


class SnailMethod(MetaTemplate):

    def __init__(self, backbone, n_way, n_support):
        super(SnailMethod, self).__init__(backbone, n_way, n_support, change_way=False)
        self.snail_model = SnailModel(backbone, n_way, n_support)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        return self.snail_model(x, y_query)


    def set_forward(self, x):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_support + self.n_query))
        y = Variable(y.cuda())

        x, y, last_targets = self.batch_for_few_shot(x, y)
        model_output = self.snail_model(x, y)

        # TODO: why 3 dimensions and not 2?
        return model_output[:, -1, :], last_targets
    

    def set_forward_loss(self, x):
        # TODO it's strange that we don't use the n_support at some point
        # TODO: check if the labels are correct or if we need to
        # pass them as argument
        last_model, last_targets = self.set_forward(x)
        return self.criterion(last_model, last_targets)

    # def test_loop(self, val_loader, return_std=None):
    # 


    def labels_to_one_hot(self, labels):
        labels = labels.cpu().numpy()
        unique = np.unique(labels)
        map = {label:idx for idx, label in enumerate(unique)}
        idxs = [map[labels[i]] for i in range(labels.size)]
        one_hot = np.zeros((labels.size, unique.size))
        one_hot[np.arange(labels.size), idxs] = 1
        return one_hot, idxs
    
    
    def batch_for_few_shot(self, x, y):
        """
            Convert each labels in the batch to a one-hot vector
            and concatenate it with the input.
            Return: x: (batch_size * seq_size, num_channels, height, width)
                    y: (batch_size * seq_size, num_cls)
                    last_targets: (batch_size), the index of the label that we want to predict
        """
        # seq_size = self.n_way * self.n_support + 1
        seq_size = self.n_way * (self.n_support + self.n_query)
        one_hots = []
        last_targets = []
        print(f"Batch size: {4}, seq_size: {seq_size}", flush=True)
        print(f"Y shape: {y.shape}", flush=True)
        print(f"Y content: {y}", flush=True)
        for i in range(4-1): # TODO use batch_size
            print(f"Batch size: {4}, i: {i}, slice: [{i * seq_size}:{(i + 1) * seq_size}]", flush=True)
            one_hot, idxs = self.labels_to_one_hot(y[i * seq_size : (i + 1) * seq_size])
            print(f"one_hot shape: {one_hot.shape}", flush=True)
            print(f"idxs shape: {len(idxs)}, content: {idxs}", flush=True)
            one_hots.append(one_hot)
            last_targets.append(idxs[-1])

        # last_targets: the index of the label that we want to predict.
        # we only take the last one of each iter as we provide all the
        # previous ones as input to the model
        last_targets = Variable(torch.Tensor(last_targets).long())

        # create a matrix of one-hot vectors
        one_hots = [torch.Tensor(temp) for temp in one_hots]
        y = torch.cat(one_hots, dim=0)
        x, y = Variable(x), Variable(y)

        # put everything on cuda
        x, y = x.cuda(), y.cuda()
        last_targets = last_targets.cuda()

        return x, y, last_targets