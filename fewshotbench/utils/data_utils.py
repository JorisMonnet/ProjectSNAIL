import random

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


class SimpleHDF5Dataset:
    def __init__(self, file_handle=None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
        # print('here')

    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i, :]), int(self.all_labels[i])

    def __len__(self):
        return self.total


def init_loader(filename):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    # labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:
        feats = np.delete(feats, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)

    class_list = np.unique(np.array(labels)).tolist()
    inds = range(len(labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append(feats[ind])

    return cl_data_file


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)

    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


def get_features(model, data_loader):
    save_features(model, data_loader, '/tmp/features.hdf5')
    cl_data_file = init_loader('/tmp/features.hdf5')
    return cl_data_file


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    norm_x = torch.norm(xx, 2)
    norm_y = torch.norm(yy, 2)

    if norm_x == 0 or norm_y == 0:
        return 0
    else:
        return torch.sum(xx * yy) / (norm_x * norm_y)


def plot(figname, x, y1, y2=None):
    plt.clf()

    plt.scatter(x, y1, label='y1', color='blue')

    if y2 is not None:
        plt.scatter(x, y2, label='y2', color='red')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Show the plot
    plt.savefig(figname)
