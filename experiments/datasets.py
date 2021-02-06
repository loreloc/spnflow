import os
import csv
import h5py
import numpy as np

BINARY_DATASETS = [
    'accidents',
    'ad',
    'baudio',
    'bbc',
    'bnetflix',
    'book',
    'c20ng',
    'cr52',
    'cwebkb',
    'dna',
    'jester',
    'kosarek',
    'msweb',
    'nltcs',
    'plants',
    'pumsb_star',
    'tmovie',
    'tretail'
]

CONTINUOUS_DATASETS = [
    'power',
    'gas',
    'hepmass',
    'miniboone',
    'bsds300'
]

VISION_DATASETS = [
    'mnist'
]


def csv_to_numpy(filepath, sep=',', type='int8'):
    with open(filepath, "r") as file:
        reader = csv.reader(file, delimiter=sep)
        dataset = np.array(list(reader)).astype(type)
        return dataset


def load_binary_dataset(root, name):
    directory = os.path.join(root, 'binary', name)
    data_train = csv_to_numpy(os.path.join(directory, name + '.train.data'))
    data_valid = csv_to_numpy(os.path.join(directory, name + '.valid.data'))
    data_test = csv_to_numpy(os.path.join(directory, name + '.test.data'))
    return data_train, data_valid, data_test


def load_continuous_dataset(root, name, standardize=True):
    filepath = os.path.join(root, 'continuous', name + '.h5')
    with h5py.File(filepath, 'r') as file:
        data_train = file['train'][:]
        data_valid = file['valid'][:]
        data_test = file['test'][:]
        if standardize:
            data_train, data_valid, data_test = dataset_standardize(data_train, data_valid, data_test)
            data_train = data_train.astype(np.float32)
            data_valid = data_valid.astype(np.float32)
            data_test = data_test.astype(np.float32)
        return data_train, data_valid, data_test


def load_vision_dataset(root, name, unsupervised=True, dequantize=True, standardize=True, flatten=False):
    filepath = os.path.join(root, 'vision', name + '.h5')
    with h5py.File(filepath, 'r') as file:
        data_train = file['train']
        data_valid = file['valid']
        data_test = file['test']
        image_train, label_train = data_train['image'][:], data_train['label'][:]
        image_valid, label_valid = data_valid['image'][:], data_valid['label'][:]
        image_test, label_test = data_test['image'][:], data_test['label'][:]
        if dequantize:
            image_train = dataset_dequantize(image_train)
            image_valid = dataset_dequantize(image_valid)
            image_test = dataset_dequantize(image_test)
        if standardize:
            image_train, image_valid, image_test = dataset_standardize(image_train, image_valid, image_test)
        image_train = image_train.astype(np.float32)
        image_valid = image_valid.astype(np.float32)
        image_test = image_test.astype(np.float32)
        label_train = label_train.astype(np.int64)
        label_valid = label_valid.astype(np.int64)
        label_test = label_test.astype(np.int64)
        if flatten:
            image_train = dataset_flatten(image_train)
            image_valid = dataset_flatten(image_valid)
            image_test = dataset_flatten(image_test)
        if unsupervised:
            return image_train, image_valid, image_test
        else:
            return (image_train, label_train), (image_valid, label_valid), (image_test, label_test)


def dataset_flatten(data):
    return data.reshape([len(data), -1])


def dataset_dequantize(data):
    return (data + np.random.rand(*data.shape)) / 256.0


def dataset_standardize(data_train, data_valid, data_test):
    data_joint = np.vstack([data_train, data_valid])
    mu = np.mean(data_joint, axis=0)
    sigma = np.std(data_joint, axis=0)
    data_train = (data_train - mu) / sigma
    data_valid = (data_valid - mu) / sigma
    data_test = (data_test - mu) / sigma
    return data_train, data_valid, data_test
