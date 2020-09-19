import os
import math
import torch
import torchvision
import numpy as np
from spnflow.torch.transforms import Flatten, Dequantize, Logit, Delogit, Reshape


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


def load_vision_dataset(root, name, supervised=False, flatten=False):
    # Get the dataset classes
    supervised_class, unsupervised_class = get_vision_dataset_classes(name)

    # Get the transform
    transform, _ = get_vision_dataset_transforms(name, supervised, flatten)

    # Load and split the dataset
    dataset_class = supervised_class if supervised else unsupervised_class
    data_train = dataset_class(root, train=True, transform=transform, download=True)
    data_test = dataset_class(root, train=False, transform=transform, download=True)
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    return data_train, data_val, data_test


def get_vision_dataset_transforms(name, supervised=False, flatten=False):
    # Get the image size of the dataset
    image_size = get_vision_dataset_image_size(name)

    # Build the forward transform
    transform = [torchvision.transforms.ToTensor()]
    if supervised:
        mean, stddev = get_vision_dataset_mean_stddev(name)
        transform.append(torchvision.transforms.Normalize(mean, stddev))
    else:
        transform.append(Dequantize(1.0 / 256.0))
        transform.append(Logit())
    transform.append(Flatten() if flatten else Reshape(*image_size))
    transform = torchvision.transforms.Compose(transform)

    # Build the image transform
    image_transform = []
    if not supervised:
        image_transform.append(Delogit())
    image_transform.append(Reshape(*image_size))
    image_transform = torchvision.transforms.Compose(image_transform)

    return transform, image_transform


def get_vision_dataset_classes(name):
    if name == 'mnist':
        return (SupervisedMNIST, UnsupervisedMNIST)
    else:
        raise ValueError


def get_vision_dataset_image_size(name):
    if name == 'mnist':
        return (1, 28, 28)
    else:
        raise ValueError


def get_vision_dataset_n_classes(name):
    if name == 'mnist':
        return 10
    else:
        raise ValueError


def get_vision_dataset_n_features(name):
    return np.prod(get_vision_dataset_image_size(name))


def get_vision_dataset_mean_stddev(name):
    if name == 'mnist':
        return (0.1307, 0.3081)
    else:
        raise ValueError


def compute_mean_quantiles(data, n_quantiles, transform=None):
    # Apply the transform, if specified
    if transform:
        data = torch.stack(list(map(transform, data.numpy())), dim=0)

    # Split the dataset in quantiles regions
    n_samples = data.size(0)
    data, indices = torch.sort(data, dim=0)
    section_quantiles = [math.floor(n_samples / n_quantiles)] * n_quantiles
    section_quantiles[-1] += n_samples % n_quantiles
    values_per_quantile = torch.split(data, section_quantiles, dim=0)

    # Compute the mean quantiles
    mean_per_quantiles = [torch.mean(x, dim=0) for x in values_per_quantile]
    return torch.stack(mean_per_quantiles, dim=0)
