import numpy as np
import pandas as pd
import tensorflow as tf
from spnflow.model.flow import AutoregressiveRatSpn
from experiments.utils import log_loss


def load_power_dataset(rand_state=None):
    # Create the random state, if necessary
    if rand_state is None:
        rand_state = np.random.RandomState(42)

    # Load the dataset and remove useless features
    data = np.load('datasets/power/data.npy')
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    rand_state.shuffle(data)

    # Add noise to the dataset
    n_samples, n_features = data.shape
    gap_noise = 0.001 * rand_state.rand(n_samples, 1)
    grp_noise = 0.001 * rand_state.rand(n_samples, 1)
    voltage_noise = 0.01 * rand_state.rand(n_samples, 1)
    global_intensity_noise = 0.1 * rand_state.rand(n_samples, 1)
    sm_noise = rand_state.rand(n_samples, 1)
    time_noise = np.zeros((n_samples, 1))
    noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
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

    return data_train, data_val, data_test


def experiment_power(rand_state):
    # Load the dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)

    # Build the model
    model = AutoregressiveRatSpn(
        depth=2,
        n_batch=4,
        n_sum=8,
        n_repetitions=8,
        optimize_scale=True,
        n_mafs=5,
        hidden_units=[128, 128],
        activation='relu',
        regularization=1e-6,
        rand_state=rand_state
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=log_loss)

    # Fit the model
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    model.fit(
        x=data_train,
        y=np.zeros((data_train.shape[0], 0), dtype=np.float32),
        validation_data=(data_val, np.zeros((data_val.shape[0], 0), dtype=np.float32)),
        epochs=100, batch_size=512,
        callbacks=[early_stopping],
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mean_log_likelihood = np.mean(y_pred)

    return mean_log_likelihood, early_stopping.stopped_epoch


if __name__ == '__main__':
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Create a results data frame
    df = pd.DataFrame(columns=['epochs', 'log-likelihood'])

    # Run the experiments
    n_experiments = 10
    for i in range(n_experiments):
        log_likelihood, epochs = experiment_power(rand_state)
        df.loc[i] = [epochs, log_likelihood]

    # Save the results data frame as CSV
    df.to_csv('results/power.csv')
