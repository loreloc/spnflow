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
            data_joint = np.vstack([data_train, data_valid])
            mu = np.mean(data_joint, axis=0)
            sigma = np.std(data_joint, axis=0)
            data_train = (data_train - mu) / sigma
            data_valid = (data_valid - mu) / sigma
            data_test = (data_test - mu) / sigma
            data_train = data_train.astype(np.float32)
            data_valid = data_valid.astype(np.float32)
            data_test = data_test.astype(np.float32)
        return data_train, data_valid, data_test
