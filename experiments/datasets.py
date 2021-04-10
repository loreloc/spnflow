import os
import csv
import h5py
import numpy as np

BINARY_DATASETS = [
    '20newsgroup',
    'accidents',
    'ad',
    'audio',
    'bbc',
    'book',
    'dna',
    'eachmovie',
    'jester',
    'kddcup2000',
    'kosarek',
    'msnbc',
    'msweb',
    'netflix',
    'nltcs',
    'plants',
    'pumsbstar',
    'retail',
    'reuters52',
    'webkb'
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


def csv_to_numpy(filepath, sep=',', dtype=np.uint8):
    with open(filepath, "r") as file:
        reader = csv.reader(file, delimiter=sep)
        dataset = np.array(list(reader)).astype(dtype)
        return dataset


def load_binary_dataset(root, name):
    directory = os.path.join(root, 'binary', name)
    data_train = csv_to_numpy(os.path.join(directory, name + '.train.data'))
    data_valid = csv_to_numpy(os.path.join(directory, name + '.valid.data'))
    data_test = csv_to_numpy(os.path.join(directory, name + '.test.data'))
    return data_train, data_valid, data_test


def load_continuous_dataset(root, name):
    filepath = os.path.join(root, 'continuous', name + '.h5')
    with h5py.File(filepath, 'r') as file:
        data_train = file['train'][:]
        data_valid = file['valid'][:]
        data_test = file['test'][:]
        return data_train, data_valid, data_test


def load_vision_dataset(root, name, unsupervised=True):
    filepath = os.path.join(root, 'vision', name + '.h5')
    with h5py.File(filepath, 'r') as file:
        data_train = file['train']
        data_valid = file['valid']
        data_test = file['test']
        image_train = data_train['image'][:]
        image_valid = data_valid['image'][:]
        image_test = data_test['image'][:]
        if len(image_train.shape[1:]) == 2:
            image_train = np.expand_dims(image_train, axis=1)
            image_valid = np.expand_dims(image_valid, axis=1)
            image_test = np.expand_dims(image_test, axis=1)
        if unsupervised:
            return image_train, image_valid, image_test
        label_train = data_train['label'][:].astype(np.int64)
        label_valid = data_valid['label'][:].astype(np.int64)
        label_test = data_test['label'][:].astype(np.int64)
        return (image_train, label_train), (image_valid, label_valid), (image_test, label_test)
