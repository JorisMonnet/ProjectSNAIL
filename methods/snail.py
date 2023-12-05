import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
import math
from backbones import AttentionBlock, TCBlock

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class SnailMethod(MetaTemplate):

    def __init__(self, backbone, n_way, n_support, num_channels):
        super(SnailMethod, self).__init__(backbone, n_way, n_support, change_way=False)

        # TODO

    def forward(self, x, labels):


    def set_forward(self, x, y):
        seq_size = self.N * self.K + 1
        one_hots = []
        last_targets = []

        # prepare the sequence with the number of examples per class
        for i in range(self.batch_size):
            one_hot, idxs = self.labels_to_one_hot(y[i * seq_size: (i + 1) * seq_size])
            one_hots.append(one_hot)
            last_targets.append(idxs[-1])

        last_targets = torch.Tensor(last_targets).long().to(self.device)

        one_hots = [torch.Tensor(temp).to(self.device) for temp in one_hots]
        y = torch.cat(one_hots, dim=0)
        x, y = x.to(self.device), y.to(self.device)

        return x, y, last_targets
    

    # def set_forward_adaptation(self, x, is_feature=False):
    #     pass


    def set_forward_loss(self, x):
        x, y, last_targets = self.set_forward(x, y)
        model_output = self.model(x, y)

        # TODO why 3 dimensions and not 2?
        last_model = model_output[:, -1, :]
        loss = loss_fn(last_model, last_targets)
        
        return loss


    def train_loop(self, epochs, tr_dataloader, optim, loss_fn):
        train_loss = []
        train_acc = []

        for epoch in range(epochs):
            print('=== Epoch: {} ==='.format(epoch))
            tr_iter = iter(tr_dataloader)
            self.model.train()
            self.model = self.model.to(self.device)
            for batch in tqdm(tr_iter):
                optim.zero_grad()
                x, y = batch
                
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(self.get_acc(last_model, last_targets))
            avg_loss = np.mean(train_loss[-len(tr_dataloader):])
            avg_acc = np.mean(train_acc[-len(tr_dataloader):])
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))


    def test_loop(self, val_loader, return_std=None):
        ...

    def labels_to_one_hot(opt, labels):
    if opt.cuda:
        labels = labels.cpu()
    labels = labels.numpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs