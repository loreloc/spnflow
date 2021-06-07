import os
import csv
import h5py
import numpy as np

from deeprob.torch.transforms import *


BINARY_DATASETS = [
    'nltcs',
    'msnbc',
    'kddcup2000',
    'plants',
    'audio',
    'jester',
    'netflix',
    'accidents',
    'retail',
    'pumsbstar',
    'dna',
    'kosarek',
    'msweb',
    'book',
    'eachmovie',
    'bmnist',
    'webkb'
    'reuters52',
    '20newsgroup',
    'bbc',
    'ad'
]

CONTINUOUS_DATASETS = [
    'power',
    'gas',
    'hepmass',
    'miniboone',
    'bsds300'
]

VISION_DATASETS = [
    'mnist',
    'cifar10'
]


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super(UnsupervisedDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.transform = transform

        # Compute the features shape
        if self.transform is None:
            self.shape = tuple(self.data.shape[1:])
        else:
            self.shape = tuple(self.transform(self.data[0]).shape)

    def features_size(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if self.transform is not None:
            x = self.transform(x)
        return x


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        super(SupervisedDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.int64)
        self.transform = transform

        # Compute the number of classes
        self.classes = np.unique(targets)

        # Compute the features shape
        if self.transform is None:
            self.shape = tuple(self.data.shape[1:])
        else:
            self.shape = tuple(self.transform(self.data[0]).shape)

    def num_classes(self):
        return len(self.classes)

    def features_size(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.targets[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def csv_to_numpy(filepath, sep=',', dtype=np.uint8):
    with open(filepath, "r") as file:
        reader = csv.reader(file, delimiter=sep)
        dataset = np.array(list(reader)).astype(dtype)
        return dataset


def load_binary_dataset(root, name, raw=False):
    directory = os.path.join(root, 'binary', name)

    # Load the CSV files to Numpy arrays
    data_train = csv_to_numpy(os.path.join(directory, name + '.train.data'))
    data_valid = csv_to_numpy(os.path.join(directory, name + '.valid.data'))
    data_test = csv_to_numpy(os.path.join(directory, name + '.test.data'))

    # Return raw Numpy arrays, if specified
    if raw:
        return data_train, data_valid, data_test

    # Instantiate the datasets
    data_train = UnsupervisedDataset(data_train)
    data_valid = UnsupervisedDataset(data_valid)
    data_test = UnsupervisedDataset(data_test)
    return data_train, data_valid, data_test


def load_continuous_dataset(root, name, raw=False):
    filepath = os.path.join(root, 'continuous', name + '.h5')

    # Open the h5 dataset files
    with h5py.File(filepath, 'r') as file:
        data_train = file['train'][:]
        data_valid = file['valid'][:]
        data_test = file['test'][:]

    # Return raw Numpy arrays, if specified
    if raw:
        return data_train, data_valid, data_test

    # Instantiate the standardize transform
    mean = torch.tensor(np.mean(data_train, axis=0), dtype=torch.float32)
    std = torch.tensor(np.std(data_train, axis=0), dtype=torch.float32)
    transform = Normalize(mean, std)

    # Instantiate the dataset
    data_train = UnsupervisedDataset(data_train, transform)
    data_valid = UnsupervisedDataset(data_valid, transform)
    data_test = UnsupervisedDataset(data_test, transform)
    return data_train, data_valid, data_test


def load_vision_dataset(root, name, unsupervised=True, raw=False, flatten=True, preproc='none'):
    filepath = os.path.join(root, 'vision', name + '.h5')

    # Load the h5 dataset files
    with h5py.File(filepath, 'r') as file:
        data_train = file['train']
        data_valid = file['valid']
        data_test = file['test']
        image_train = data_train['image'][:]
        image_valid = data_valid['image'][:]
        image_test = data_test['image'][:]
        label_train = data_train['label'][:]
        label_valid = data_valid['label'][:]
        label_test = data_test['label'][:]
        if len(image_train.shape[1:]) == 2:
            image_train = np.expand_dims(image_train, axis=1)
            image_valid = np.expand_dims(image_valid, axis=1)
            image_test = np.expand_dims(image_test, axis=1)

    # Return raw Numpy arrays, if specified
    if raw:
        if unsupervised:
            return image_train, image_valid, image_test
        return (image_train, label_train), (image_valid, label_valid), (image_test, label_test)

    # Build the transforms
    transform = None
    shape = torch.Size(image_train.shape[1:])
    if preproc == 'none':
        if flatten:
            transform = Flatten(shape)
    elif preproc == 'normalize':
        if flatten:
            transform = TransformList([Flatten(shape), Normalize(0.0, 255.0)])
        else:
            transform = Normalize(0.0, 255.0)
    elif preproc == 'standardize':
        mean = np.mean(image_train)
        std = np.std(image_train)
        if flatten:
            transform = TransformList([Flatten(shape), Normalize(mean, std)])
        else:
            transform = Normalize(mean, std)
    else:
        raise NotImplementedError('Unknown preprocessing method {}'.format(preproc))

    if unsupervised:
        # Instantiate unsupervised datasets
        image_train = UnsupervisedDataset(image_train, transform)
        image_valid = UnsupervisedDataset(image_valid, transform)
        image_test = UnsupervisedDataset(image_test, transform)
        return image_train, image_valid, image_test

    # Instantiate supervised datasets
    data_train = SupervisedDataset(image_train, label_train, transform)
    data_valid = SupervisedDataset(image_valid, label_valid, transform)
    data_test = SupervisedDataset(image_test, label_test, transform)
    return data_train, data_valid, data_test
