import os
import numpy as np


def load_miniboone_dataset(rand_state):
    # Load the dataset
    filepath = os.path.join(os.environ['DATAPATH'], 'datasets/miniboone/data.npy')
    data = np.load(filepath)
    rand_state.shuffle(data)

    # Split the dataset in train, validation and test set
    n_test = int(0.1 * data.shape[0])
    data_test = data[-n_test:]
    data = data[0:-n_test]
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    n_val = int(0.1 * data.shape[0])
    data_val = data[-n_val:]
    data_train = data[0:-n_val]

    # Normalize the data
    data_train = (data_train - mu) / sigma
    data_val = (data_val - mu) / sigma
    data_test = (data_test - mu) / sigma
    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')

    return data_train, data_val, data_test
