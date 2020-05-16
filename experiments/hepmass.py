import numpy as np
import pandas as pd
from collections import Counter


def load_hepmass_dataset(rand_state):
    # Load the train and test dataset
    data_train = pd.read_csv('datasets/hepmass/1000_train.csv', index_col=False)
    data_test = pd.read_csv('datasets/hepmass/1000_test.csv', index_col=False)

    # Get rid of discrete features
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    # Shuffle the train data
    data_train = data_train.to_numpy()
    data_test = data_test.to_numpy()
    rand_state.shuffle(data_train)

    # Remove useless features
    i = 0
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    # Split the dataset in train, validation and test set
    mu = np.mean(data_train, axis=0)
    sigma = np.std(data_train, axis=0)
    n_val = int(0.1 * data_train.shape[0])
    data_val = data_train[-n_val:]
    data_train = data_train[0:-n_val]

    # Normalize the data
    data_train = (data_train - mu) / sigma
    data_val = (data_val - mu) / sigma
    data_test = (data_test - mu) / sigma
    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')

    return data_train, data_val, data_test
