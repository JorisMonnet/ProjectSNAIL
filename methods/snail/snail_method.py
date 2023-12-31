import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from methods.snail.snail_model import SnailModel


class SnailMethod(MetaTemplate):
    """
        This class is a wrapper around the SnailModel class.
        It implements the methods from MetaTemplate that are specific to the SNAIL method.
    """

    def __init__(self, backbone, n_way, n_support, architecture):
        super(SnailMethod, self).__init__(backbone, n_way, n_support, change_way=False)
        self.snail_model = SnailModel(backbone, n_way, n_support, architecture)
        self.criterion = nn.CrossEntropyLoss()  # softmax is applied in the loss
        self.n_query_snail = 1

    def forward(self, x):
        """
        Forward pass of the model
        :param x: (batch_size, seq_size, num_channels, height, width)
        :return: model_output
        """
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        return self.snail_model(x, y_query)

    def set_forward(self, x, y):
        """
            Run one episode of the SNAIL method and return the target output
            :param x: (batch_size, seq_size, num_channels, height, width)
            :param y: (batch_size, seq_size, num_cls)
            :return: model_output: (batch_size, num_cls)
                        targets: (batch_size)
        """
        x = x.reshape(-1, *x.size()[2:])

        x, y, targets = self.batch_for_few_shot(x, y)

        model_output = self.snail_model(x, y)

        return model_output[:, -1, :], targets

    def set_forward_loss(self, x, y):
        """
            Compute the SNAIL method loss over one episode
            :param x: (batch_size, seq_size, num_channels, height, width)
            :param y: (batch_size, seq_size, num_cls)
            :return: loss
        """
        model_preds, targets = self.set_forward(x, y)

        # get the last model
        targets = targets.view(-1)

        return self.criterion(model_preds, targets)

    def compute_accuracy(self, model_preds, targets):
        """
            Compute the accuracy of the model
            :param model_preds: (batch_size, num_cls)
            :param targets: (batch_size)
            :return: accuracy
        """
        targets = targets.view(-1)

        # Get the predicted classes by finding the index of the maximum logit
        _, pred_classes = torch.max(model_preds, 1)

        return (pred_classes == targets).sum().item() / targets.size(0) * 100

    def train_loop(self, epoch, train_loader, optimizer):
        """
            Train the model for one epoch
            :param epoch: int
            :param train_loader: loader
            :param optimizer: optimizer
            :return: None
        """
        print_freq = 10

        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            optimizer.zero_grad()

            # put x and y in cuda
            x, y = x.cuda(), y.cuda()

            loss = self.set_forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({"loss": avg_loss / float(i + 1)})

    def test_loop(self, test_loader, record=None, return_std=False):
        """
            Test the model on the test set
            :param test_loader: loader for the test set
            :param record: not used
            :param return_std: bool, whether to return the standard deviation of the accuracy
            :return: test accuracy
        """
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            model_preds, targets = self.set_forward(x, y)
            acc = self.compute_accuracy(model_preds, targets)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def labels_to_one_hot(self, labels):
        """
            Convert a vector of labels to a matrix of one-hot vectors
            :param labels: (batch_size)
            :return: one_hot: (batch_size, num_cls)
        """
        labels = labels.cpu().numpy()
        unique, unique_idx = np.unique(labels, return_index=True)
        unique = unique[np.argsort(unique_idx)]

        label_mapper = {label: idx for idx, label in enumerate(unique)}
        idxs = [label_mapper[labels[i]] for i in range(labels.size)]
        one_hot = np.zeros((labels.size, unique.size))
        one_hot[np.arange(labels.size), idxs] = 1
        return one_hot, idxs

    def split_support_query_labels(self, original_tensor):
        """
            Split the labels of a sequence in the fewshotbench format
            :param original_tensor: (batch_size * seq_size)
            :return: support_tensor: (batch_size * n_way * n_support)
                     query_tensor: (batch_size * n_way * n_query)
        """
        # Reshape the tensor to separate each 'way'
        reshaped_tensor = original_tensor.view(self.n_way, self.n_support + self.n_query)

        # Extract support and query tensors
        support_tensor = reshaped_tensor[:, :self.n_support]
        query_tensor = reshaped_tensor[:, self.n_support:]

        # put both tensors in the fewshotbench format
        # i.e. concat all supports together and all queries together
        support_tensor = support_tensor.reshape(-1)
        query_tensor = query_tensor.reshape(-1)

        assert support_tensor.size()[0] == self.n_way * self.n_support
        assert query_tensor.size()[0] == self.n_way * self.n_query

        return support_tensor, query_tensor

    def fsb_to_snail_seq_label(self, y_fsb, target_query):
        """
            Convert a sequence in the fewshotbench format to the snail format
            :param y_fsb: (batch_size * seq_size)
            :param target_query: (batch_size)
            :return: y_support: (batch_size * n_way * n_support + n_query_snail)
        """
        # split support and queries
        y_support, y_query = self.split_support_query_labels(y_fsb)

        # sample n_query randomly from the query set and concatenate them to the support set
        targets_labels = y_query[target_query]

        # put the labels and the last targets together
        y_support = torch.cat((y_support, targets_labels), dim=0)
        assert y_support.size()[0] == self.n_way * self.n_support + self.n_query_snail

        return y_support

    def split_support_query_data(self, original_tensor):
        """
            Split the data of a sequence in the fewshotbench format
            :param original_tensor: (batch_size * seq_size, num_channels, height, width)
            :return: support_tensor: (batch_size * n_way * n_support, num_channels, height, width)
        """
        # Reshape the tensor to separate each 'way'
        reshaped_tensor = original_tensor.view(self.n_way, self.n_support + self.n_query, *original_tensor.size()[1:])

        # Extract support and query tensors
        support_tensor = reshaped_tensor[:, :self.n_support]
        query_tensor = reshaped_tensor[:, self.n_support:]

        # put both tensors in the fewshotbench format
        # i.e. concat all supports together and all queries together
        support_tensor = support_tensor.reshape(-1, *original_tensor.size()[1:])
        query_tensor = query_tensor.reshape(-1, *original_tensor.size()[1:])

        assert support_tensor.size()[0] == self.n_way * self.n_support
        assert query_tensor.size()[0] == self.n_way * self.n_query

        return support_tensor, query_tensor

    def fsb_to_snail_seq_data(self, x_fsb, target_query):
        """
            Convert a sequence in the fewshotbench format to the snail format
            :param x_fsb: (batch_size * seq_size, num_channels, height, width)
            :param target_query: (batch_size)
            :return: x_support: (batch_size * n_way * n_support + n_query_snail, num_channels, height, width)
        """
        # split support and queries
        x_support, x_query = self.split_support_query_data(x_fsb)

        # sample n_query randomly from the query set and concatenate them to the support set
        assert x_query.size()[0] == self.n_way * self.n_query
        targets_data = x_query[target_query]

        # put the labels and the last targets together
        x_support = torch.cat((x_support, targets_data), dim=0)
        assert x_support.size()[0] == self.n_way * self.n_support + self.n_query_snail

        return x_support

    def batch_for_few_shot(self, x, y):
        """
            Convert each labels in the batch to a one-hot vector
            and concatenate it with the input.
            Return: x: (batch_size * seq_size, num_channels, height, width)
                    y: (batch_size * seq_size, num_cls)
                    targets: (batch_size), the index of the label that we want to predict
        """
        one_hots = []
        targets = []

        target_query = torch.randperm(self.n_way)[:self.n_query_snail]
        x_snail = self.fsb_to_snail_seq_data(x, target_query)
        y_snail = self.fsb_to_snail_seq_label(y, target_query)

        one_hot, idxs = self.labels_to_one_hot(y_snail)
        one_hots.append(one_hot)
        targets.append(idxs[-self.n_query_snail:])

        # targets: the index of the label that we want to predict.
        # we only take the last one of each iter as we provide all the
        # previous ones as input to the model
        targets = Variable(torch.Tensor(targets).long()).cuda()

        # create a matrix of one-hot vectors
        one_hots = [torch.Tensor(temp) for temp in one_hots]
        y_one_hot = torch.cat(one_hots, dim=0)
        x_snail, y_one_hot = Variable(x_snail).cuda(), Variable(y_one_hot).cuda()

        return x_snail, y_one_hot, targets
