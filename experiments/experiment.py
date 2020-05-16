import os
import sys
import numpy as np
import tensorflow as tf
from experiments.utils import log_loss
from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.mnist import load_mnist_dataset
from spnflow.model.flow import AutoregressiveRatSpn


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=1, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'made', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=5, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'maf5', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=10, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'maf10', model, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=1, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'made', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=5, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'maf5', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=10, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'maf10', model, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'made', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'maf5', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=10, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'maf10', model, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'made', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'maf5', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=10, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'maf10', model, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=1, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'made', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=5, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'maf5', model, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=10, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'maf10', model, data_train, data_val, data_test)


def collect_results(dataset, info, model, data_train, data_val, data_test):
    # Run the experiment and get the results
    mu_ll, sigma_ll = experiment_log_likelihood(model, data_train, data_val, data_test)

    # Save the results to file
    with open(os.path.join('results', dataset + '_' + info + '.txt'), 'w') as file:
        file.write(dataset + '\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Std. Log-Likelihood: ' + str(sigma_ll) + '\n')


def experiment_log_likelihood(model, data_train, data_val, data_test):
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=log_loss)

    # Fit the model
    model.fit(
        x=data_train,
        y=np.zeros((data_train.shape[0], 0), dtype=np.float32),
        validation_data=(data_val, np.zeros((data_val.shape[0], 0), dtype=np.float32)),
        epochs=200,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
    )

    # Compute the test set mean log likelihood
    y_pred = model.predict(data_test)
    mu_log_likelihood = np.mean(y_pred)
    sigma_log_likelihood = np.std(y_pred) / np.sqrt(data_test.shape[0])

    return mu_log_likelihood, sigma_log_likelihood


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataset_fn = sys.argv[1]
    if dataset_fn == 'power':
        run_experiment_power()
    elif dataset_fn == 'gas':
        run_experiment_gas()
    elif dataset_fn == 'hepmass':
        run_experiment_hepmass()
    elif dataset_fn == 'miniboone':
        run_experiment_miniboone()
    elif dataset_fn == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset_fn)
