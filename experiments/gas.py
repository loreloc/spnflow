import numpy as np
import pandas as pd
import tensorflow as tf
from spnflow.model.flow import AutoregressiveRatSpn
from experiments.utils import log_loss


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


if __name__ == '__main__':
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)

    # Build the model
    model = AutoregressiveRatSpn(
        depth=2,
        n_batch=4,
        n_sum=8,
        n_repetitions=8,
        n_mafs=5,
        hidden_units=[128, 128],
        activation='relu',
        regularization=1e-6,
        rand_state=rand_state
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=log_loss)

    # Fit the model
    model.fit(
        x=data_train,
        y=np.zeros((data_train.shape[0], 0), dtype=np.float32),
        validation_data=(data_val, np.zeros((data_val.shape[0], 0), dtype=np.float32)),
        epochs=200, batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mu_log_likelihood = np.mean(y_pred)
    sigma_log_likelihood = np.std(y_pred) / np.sqrt(data_test.shape[0])

    # Save the results to file
    with open('results/gas.txt', 'w') as file:
        file.write('Gas;\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_log_likelihood) + '\n')
        file.write('Std. Log-Likelihood: ' + str(sigma_log_likelihood) + '\n')
