import os
import torch
import torchvision
import numpy as np

from spnflow.torch.transforms import Flatten, Dequantize, Reshape


class SupervisedDataset(torch.utils.data.Dataset):
    """Supervised vision dataset"""
    def __init__(self, root, dataset_class, **kwargs):
        self.dataset = dataset_class(root, **kwargs)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class UnsupervisedDataset(torch.utils.data.Dataset):
    """unsupervised vision dataset"""
    def __init__(self, root, dataset_class, **kwargs):
        self.dataset = dataset_class(root, **kwargs)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x

    def __len__(self):
        return len(self.dataset)


def load_dataset(root, name, rand_state, standardize=True):
    # Load the dataset
    data_train = np.load(os.path.join(root, name, 'train.npy'))
    data_test = np.load(os.path.join(root, name, 'test.npy'))
    rand_state.shuffle(data_train)

    if standardize:
        # Standardize the data
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


def load_vision_dataset(root, name, supervised=False, dequantize=False, standardize=False, flatten=False):
    # Get the transform
    transform = get_vision_dataset_transform(name, dequantize, standardize, flatten)

    # Load the dataset
    dataset_class = get_vision_dataset_class(name)
    if supervised:
        data_train = SupervisedDataset(root, dataset_class, train=True, transform=transform, download=True)
        data_test = SupervisedDataset(root, dataset_class, train=False, transform=transform, download=True)
    else:
        data_train = UnsupervisedDataset(root, dataset_class, train=True, transform=transform, download=True)
        data_test = UnsupervisedDataset(root, dataset_class, train=False, transform=transform, download=True)

    # Split the train set in train set and validation set
    n_val = int(0.1 * len(data_train))
    n_train = len(data_train) - n_val
    data_train, data_val = torch.utils.data.random_split(data_train, [n_train, n_val])
    return data_train, data_val, data_test


def get_vision_dataset_transform(name, dequantize=False, standardize=False, flatten=False):
    # Get the image size of the dataset
    image_size = get_vision_dataset_image_size(name)

    # Build the transform
    transforms = [torchvision.transforms.ToTensor()]
    if dequantize:
        transforms.append(Dequantize())
    if standardize:
        mean, stddev = get_vision_dataset_mean_stddev(name)
        transforms.append(torchvision.transforms.Normalize(mean, stddev))
    if flatten:
        transforms.append(Flatten())
    else:
        transforms.append(Reshape(image_size))
    return torchvision.transforms.Compose(transforms)


def get_vision_dataset_inverse_transform(name, standardize=False):
    # Get the image size of the dataset
    image_size = get_vision_dataset_image_size(name)

    # Build the transforms
    transforms = [Reshape(image_size)]
    if standardize:
        mean, stddev = get_vision_dataset_mean_stddev(name)
        transforms.append(torchvision.transforms.Normalize(-mean / stddev, 1.0 / stddev))
    return torchvision.transforms.Compose(transforms)


def get_vision_dataset_class(name):
    if name == 'mnist':
        return torchvision.datasets.MNIST
    elif name == 'cifar10':
        return torchvision.datasets.CIFAR10
    else:
        raise ValueError


def get_vision_dataset_image_size(name):
    if name == 'mnist':
        return 1, 28, 28
    elif name == 'cifar10':
        return 3, 32, 32
    else:
        raise ValueError


def get_vision_dataset_n_classes(name):
    if name == 'mnist':
        return 10
    elif name == 'cifar10':
        return 10
    else:
        raise ValueError


def get_vision_dataset_n_features(name):
    return np.prod(get_vision_dataset_image_size(name))


def get_vision_dataset_mean_stddev(name):
    if name == 'mnist':
        return np.array(0.1307), np.array(0.3081)
    elif name == 'cifar10':
        return np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])
    else:
        raise ValueError
