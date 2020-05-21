import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from experiments.utils import log_loss
from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.mnist import load_mnist_dataset, delogit
from spnflow.model.rat import RatSpn
from spnflow.model.flow import AutoregressiveRatSpn

EPOCHS = 500
BATCH_SIZE = 128
PATIENCE = 30
LR_SPN = 1e-3
LR_MADE = 1e-3
LR_MAF = 1e-4


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=1, n_batch=8, n_sum=8, n_repetitions=8, rand_state=rand_state)
    collect_results('power', 'spn', model, LR_SPN, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=1, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'made', model, LR_MADE, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=5, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'maf5', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=10, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'maf10', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=1, n_batch=8, n_sum=8, n_repetitions=8, rand_state=rand_state)
    collect_results('gas', 'spn', model, LR_SPN, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=1, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'made', model, LR_MADE, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=5, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'maf5', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=8,
        n_mafs=10, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'maf10', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=2, n_batch=16, n_sum=16, n_repetitions=16, rand_state=rand_state)
    collect_results('hepmass', 'spn', model, LR_SPN, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'made', model, LR_MADE, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'maf5', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=10, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'maf10', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=2, n_batch=16, n_sum=16, n_repetitions=16, rand_state=rand_state)
    collect_results('miniboone', 'spn', model, LR_SPN, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'made', model, LR_MADE, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'maf5', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=16, n_sum=16, n_repetitions=16,
        n_mafs=10, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'maf10', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=3, n_batch=16, n_sum=16, n_repetitions=32, rand_state=rand_state)
    collect_results('mnist', 'spn', model, LR_SPN, data_train, data_val, data_test)
    collect_samples('mnist', 'spn', model, 20, post_fn=delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=1, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'made', model, LR_MADE, data_train, data_val, data_test)
    collect_samples('mnist', 'made', model, 20, post_fn=delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=5, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'maf5', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('mnist', 'maf5', model, 20, post_fn=delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=10, hidden_units=[1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'maf10', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('mnist', 'maf10', model, 20, post_fn=delogit)


def collect_results(dataset, info, model, lr, data_train, data_val, data_test):
    # Run the experiment and get the results
    mu_ll, sigma_ll = experiment_log_likelihood(model, lr, data_train, data_val, data_test)

    # Save the results to file
    with open(os.path.join('results', dataset + '_' + info + '.txt'), 'w') as file:
        file.write(dataset + '\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two Std. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')


def collect_samples(dataset, info, model, n_samples, post_fn=None):
    # Get some samples
    samples = model.sample(n_samples)

    # Post process the samples
    if post_fn != None:
        samples = post_fn(samples)
    _, n_features = samples.shape
    img_size = int(np.sqrt(n_features))

    # Plot the samples
    fig, axs = plt.subplots(1, n_samples, figsize=(n_samples, 1))
    fig.subplots_adjust(top=1.0, bottom=0.0, right=1.0, left=0.0, wspace=0.0, hspace=0.0)
    for i in range(n_samples):
        axs[i].axis('off')
        axs[i].imshow(np.reshape(samples[i], (img_size, img_size)), cmap='gray', interpolation='nearest')
    fig.savefig(os.path.join('results', dataset + '_' + info + '.png'))


def experiment_log_likelihood(model, lr, data_train, data_val, data_test):
    # Instantiate the optimizer
    optimizer = tf.keras.optimizers.Adam(lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss=log_loss)

    # Fit the model
    model.fit(
        x=data_train,
        y=np.zeros((data_train.shape[0], 0), dtype=np.float32),
        validation_data=(data_val, np.zeros((data_val.shape[0], 0), dtype=np.float32)),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE)]
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
