import numpy as np


def load_power_dataset(rand_state):
    # Load the dataset and remove useless features
    data = np.load('datasets/power/data.npy')
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    rand_state.shuffle(data)

    # Add noise to the dataset
    n_samples, n_features = data.shape
    gap_noise = 0.001 * rand_state.rand(n_samples, 1)
    voltage_noise = 0.01 * rand_state.rand(n_samples, 1)
    sm_noise = rand_state.rand(n_samples, 3)
    time_noise = np.zeros((n_samples, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data += noise

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
