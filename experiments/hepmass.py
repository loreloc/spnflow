import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from spnflow.model.flow import AutoregressiveRatSpn
from experiments.utils import log_loss


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


if __name__ == '__main__':
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)

    # Build the model
    model = AutoregressiveRatSpn(
        depth=2,
        n_batch=4,
        n_sum=8,
        n_repetitions=8,
        optimize_scale=True,
        n_mafs=5,
        hidden_units=[512, 512],
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
        epochs=200, batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mu_log_likelihood = np.mean(y_pred)
    sigma_log_likelihood = np.std(y_pred) / np.sqrt(data_test.shape[0])

    # Save the results to file
    with open('results/hepmass.txt', 'w') as file:
        file.write('Hepmass;\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_log_likelihood) + '\n')
        file.write('Std. Log-Likelihood: ' + str(sigma_log_likelihood) + '\n')
