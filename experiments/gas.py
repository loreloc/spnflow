import numpy as np
import pandas as pd


def load_gas_dataset(rand_state=None):
    # Load the dataset and remove useless features
    data = pd.read_pickle('datasets/gas/ethylene_CO.pickle')
    data.drop('Meth', axis=1, inplace=True)
    data.drop('Eth', axis=1, inplace=True)
    data.drop('Time', axis=1, inplace=True)
    uninformative_features = (data.corr() > 0.98).to_numpy().sum(axis=1)
    while np.any(uninformative_features > 1):
        col = np.where(uninformative_features > 1)[0][0]
        col_name = data.columns[col]
        data.drop(col_name, axis=1, inplace=True)
        uninformative_features = (data.corr() > 0.98).to_numpy().sum(axis=1)
    data = data.to_numpy()
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    rand_state.shuffle(data)

    # Split the dataset in train, validation and test set
    n_test = int(0.1 * data.shape[0])
    data_test = data[-n_test:]
    data = data[0:-n_test]
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
