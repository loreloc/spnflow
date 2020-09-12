import os
import math
import torch
import torchvision
import numpy as np


class SupervisedMNIST(torchvision.datasets.MNIST):
    """Supervised MNIST"""
    def __init__(self, *args, **kwargs):
        super(SupervisedMNIST, self).__init__(*args, **kwargs)


class UnsupervisedMNIST(torchvision.datasets.MNIST):
    """Unsupervised MNIST"""
    def __init__(self, *args, **kwargs):
        super(UnsupervisedMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        x, y = super(UnsupervisedMNIST, self).__getitem__(index)
        return x

    def mean_quantiles(self, n_quantiles):
        return compute_mean_quantiles(self.data, n_quantiles, self.transform)


def load_dataset(root, name, rand_state, normalize=True):
    # Load the dataset
    data_train = np.load(os.path.join(root, name, 'train.npy'))
    data_test = np.load(os.path.join(root, name, 'test.npy'))
    rand_state.shuffle(data_train)

    if normalize:
        # Normalize the data
        mu = np.mean(data_train, axis=0)
        sigma = np.std(data_train, axis=0)
        data_train = (data_train - mu) / sigma
        data_test = (data_test - mu) / sigma
        data_train = data_train.astype('float32')
        data_test = data_test.astype('float32')

    # Split train data to get the validation data
    n_val = int(0.1 * len(data_train))
    data_val = data_train[-n_val:]
    data_train = data_train[0:-n_val]

    return data_train, data_val, data_test


def load_supervised_mnist(root, transform=None):
    # Load and split the MNIST dataset (supervised setting)
    data_train = SupervisedMNIST(root, train=True, transform=transform, download=True)
    data_test = SupervisedMNIST(root, train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    return data_train, data_val, data_test


def load_unsupervised_mnist(root, transform=None):
    # Load and split the MNIST dataset (unsupervised setting)
    data_train = UnsupervisedMNIST(root, train=True, transform=transform, download=True)
    data_test = UnsupervisedMNIST(root, train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    return data_train, data_val, data_test


def compute_mean_quantiles(data, n_quantiles, transform=None):
    if transform:
        data = torch.stack(list(map(transform, data.numpy())), dim=0)
    n_samples = data.size(0)
    data, indices = torch.sort(data, dim=0)
    section_quantiles = [math.floor(n_samples / n_quantiles)] * n_quantiles
    section_quantiles[-1] += n_samples % n_quantiles
    values_per_quantile = torch.split(data, section_quantiles, dim=0)
    mean_per_quantiles = [torch.mean(x, dim=0) for x in values_per_quantile]
    return torch.stack(mean_per_quantiles, dim=0)
