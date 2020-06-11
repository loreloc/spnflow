import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.bsds300 import load_bsds300_dataset
from experiments.mnist import load_mnist_dataset
from experiments.mnist import delogit as mnist_delogit
from experiments.mnist import plot as mnist_plot
from experiments.cifar10 import load_cifar10_dataset
from experiments.cifar10 import delogit as cifar10_delogit
from experiments.cifar10 import plot as cifar10_plot

from spnflow.tensorflow.model.rat import RatSpn
from spnflow.tensorflow.model.flow import AutoregressiveRatSpn
from spnflow.tensorflow.utils import log_loss

EPOCHS = 1000
BATCH_SIZE = 100
EPSILON = 1e-4
PATIENCE = 30
LR_RAT = 1e-3
LR_MAF = 1e-4


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=1, n_batch=8, n_sum=8, n_repetitions=16, rand_state=rand_state)
    collect_results('power', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=1, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=3, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=5, hidden_units=[128, 128], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('power', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=1, n_batch=8, n_sum=8, n_repetitions=16, rand_state=rand_state)
    collect_results('gas', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=1, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=3, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=1, n_batch=8, n_sum=8, n_repetitions=16,
        n_mafs=5, hidden_units=[128, 128], activation='tanh', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('gas', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=2, n_batch=8, n_sum=8, n_repetitions=32, rand_state=rand_state)
    collect_results('hepmass', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=3, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('hepmass', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    #model = RatSpn(n_features, depth=2, n_batch=8, n_sum=8, n_repetitions=32, rand_state=rand_state)
    #collect_results('miniboone', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=3, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('miniboone', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_bsds300_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=2, n_batch=8, n_sum=8, n_repetitions=32, rand_state=rand_state)
    collect_results('bsds300', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=1, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('bsds300', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=3, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('bsds300', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)

    model = AutoregressiveRatSpn(
        depth=2, n_batch=8, n_sum=8, n_repetitions=32,
        n_mafs=5, hidden_units=[512, 512], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('bsds300', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=3, n_batch=16, n_sum=16, n_repetitions=32, rand_state=rand_state)
    collect_results('mnist', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)
    collect_samples('mnist', 'rat-spn', model, 36, mnist_plot, mnist_delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=1, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('mnist', 'rat-spn-maf1', model, 36, mnist_plot, mnist_delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=3, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('mnist', 'rat-spn-maf3', model, 36, mnist_plot, mnist_delogit)

    model = AutoregressiveRatSpn(
        depth=3, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=5, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('mnist', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('mnist', 'rat-spn-maf5', model, 36, mnist_plot, mnist_delogit)


def run_experiment_cifar10():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the cifar10 dataset
    data_train, data_val, data_test = load_cifar10_dataset(rand_state)
    _, n_features = data_train.shape

    model = RatSpn(n_features, depth=4, n_batch=16, n_sum=16, n_repetitions=32, rand_state=rand_state)
    collect_results('cifar10', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)
    collect_samples('cifar10', 'rat-spn', model, 36, cifar10_plot, cifar10_delogit)

    model = AutoregressiveRatSpn(
        depth=4, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=1, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('cifar10', 'rat-spn-maf1', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('cifar10', 'rat-spn-maf1', model, 36, cifar10_plot, cifar10_delogit)

    model = AutoregressiveRatSpn(
        depth=4, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=3, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('cifar10', 'rat-spn-maf3', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('cifar10', 'rat-spn-maf3', model, 36, cifar10_plot, cifar10_delogit)

    model = AutoregressiveRatSpn(
        depth=4, n_batch=16, n_sum=16, n_repetitions=32,
        n_mafs=5, hidden_units=[1024, 1024], activation='relu', regularization=1e-6,
        rand_state=rand_state
    )
    collect_results('cifar10', 'rat-spn-maf5', model, LR_MAF, data_train, data_val, data_test)
    collect_samples('cifar10', 'rat-spn-maf5', model, 36, cifar10_plot, cifar10_delogit)


def collect_results(dataset, info, model, lr, data_train, data_val, data_test):
    # Run the experiment and get the results
    history, (mu_ll, sigma_ll) = experiment_log_likelihood(model, lr, data_train, data_val, data_test)

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two Std. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')

    # Plot the training history
    filepath = os.path.join('histories', dataset + '_' + info + '.png')
    plt.xlim(0, EPOCHS)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'])
    plt.savefig(filepath)
    plt.clf()


def collect_samples(dataset, info, model, n_samples, plot_fn, post_fn=None):
    # Get some samples
    samples = model.sample(n_samples)

    # Post process the samples
    if post_fn != None:
        samples = post_fn(samples)

    # Plot the samples
    fig, axs = plt.subplots(1, n_samples, figsize=(n_samples, 1))
    fig.subplots_adjust(top=1.0, bottom=0.0, right=1.0, left=0.0, wspace=0.0, hspace=0.0)
    for i in range(n_samples):
        axs[i].axis('off')
        plot_fn(axs[i], samples[i])

    fig.savefig(os.path.join('results', dataset + '_' + info + '.png'))


def experiment_log_likelihood(model, lr, data_train, data_val, data_test):
    # Instantiate the optimizer
    optimizer = tf.keras.optimizers.Adam(lr, epsilon=EPSILON)

    # Compile the model
    model.compile(optimizer=optimizer, loss=log_loss)

    # Fit the model
    history = model.fit(
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

    return history, (mu_log_likelihood, sigma_log_likelihood)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataset = sys.argv[1]
    if dataset == 'power':
        run_experiment_power()
    elif dataset == 'gas':
        run_experiment_gas()
    elif dataset == 'hepmass':
        run_experiment_hepmass()
    elif dataset == 'miniboone':
        run_experiment_miniboone()
    elif dataset == 'bsds300':
        run_experiment_bsds300()
    elif dataset == 'mnist':
        run_experiment_mnist()
    elif dataset == 'cifar10':
        run_experiment_cifar10()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
