import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from backbones.blocks import distLinear
from methods.meta_template import MetaTemplate


class Baseline(MetaTemplate):

    def __init__(self, backbone, n_way, n_support, n_classes=1, loss='softmax', type='classification'):
        super(Baseline, self).__init__(backbone, n_way, n_support, change_way=True)
        self.feature = backbone
        self.type = type
        self.n_classes = n_classes

        if loss == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, n_classes)
            self.classifier.bias.data.fill_(0)
        elif loss == 'dist':  # Baseline ++
            self.classifier = distLinear(self.feature.final_feat_dim, n_classes)

        self.loss_type = loss  # 'softmax' #'dist'

        if self.type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.type == 'regression':
            self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def forward(self, x):
        if isinstance(x, list):
            x = [Variable(obj.cuda()) for obj in x]
        else:
            x = Variable(x.cuda())

        out = self.feature.forward(x)
        if self.classifier != None:
            scores = self.classifier.forward(out)
        return scores

    def set_forward_loss(self, x, y):
        scores = self.forward(x)
        print(scores.shape)
        if self.type == 'classification':
            y = y.long().cuda()
        else:
            y = y.cuda()

        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y)

            # if self.change_way:
            #     self.n_way = self.n_classes
            
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss/train": avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=None):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
            else:
                self.n_query = x.size(1) - self.n_support

            if self.type == "classification":
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
            else:
                # Use pearson correlation
                acc_all.append(self.correlation(x, y))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        if self.type == "classification":
            print('%d Accuracy = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        else:
            # print correlation
            print('%d Correlation = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward(self, x, y=None):
        z_support, z_query = self.parse_feature(x, is_feature=False)

        # Detach ensures we don't change the weights in main training process
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1).detach().to(self.device)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).detach().to(self.device)

        if y is None:  # Classification
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
            y_support = Variable(y_support.to(self.device))
        else:  # Regression
            y_support = y[:, :self.n_support]
            y_support = y_support.contiguous().view(self.n_way * self.n_support, -1).to(self.device)
            # y_support = y_support.contiguous().view(self.n_way * y.size(1), -1)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = distLinear(self.feat_dim, self.n_way)
        else:
            raise ValueError('Loss type not supported')

        linear_clf = linear_clf.to(self.device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)

        loss_function = self.loss_fn.to(self.device)

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                # loss.backward(retain_graph=True)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
