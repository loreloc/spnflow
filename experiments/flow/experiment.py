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

from spnflow.tensorflow.utils import log_loss
from spnflow.tensorflow.models import RatSpn, RatSpnFlow

EPOCHS = 1000
BATCH_SIZE = 100
PATIENCE = 30
LR_RAT = 5e-4
LR_FLOW = 1e-4
N_SAMPLES = 36


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 1, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 8, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [128] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [128] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [128] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [128] * 2},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [128] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [128] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [128] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [128] * 2}
    ]

    model = RatSpn(**ratspn_kwargs)
    collect_results('power', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('power', ratspn_flow_info(kwargs), model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 1, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 8, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [128] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [128] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [128] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [128] * 2},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [128] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [128] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [128] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [128] * 2}
    ]

    model = RatSpn(**ratspn_kwargs)
    collect_results('gas', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='tanh', regularization=1e-6)
        collect_results('gas', ratspn_flow_info(kwargs), model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 2, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 2},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 2}
    ]

    model = RatSpn(**ratspn_kwargs)
    collect_results('hepmass', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('hepmass', ratspn_flow_info(kwargs), model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 2, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 2},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 2}
    ]

    model = RatSpn(**ratspn_kwargs)
    collect_results('miniboone', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('miniboone', ratspn_flow_info(kwargs), model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_bsds300_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 2, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 16, 'rand_state': rand_state
    }

    # Set the parameters for the nvp conditioners
    nvp_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [512] * 2},
    ]

    # Set the parameters for the maf conditioners
    maf_kwargs = [
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [512] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [512] * 2}
    ]

    lr_rat = LR_RAT * 5e-1
    model = RatSpn(**ratspn_kwargs)
    collect_results('bsds300', 'rat-spn', model, lr_rat, data_train, data_val, data_test)

    lr_flow = LR_FLOW * 5e-1
    for kwargs in nvp_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('bsds300', ratspn_flow_info(kwargs), model, lr_flow, data_train, data_val, data_test)

    lr_flow = LR_FLOW * 2e-1
    for kwargs in maf_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('bsds300', ratspn_flow_info(kwargs), model, lr_flow, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'depth': 3, 'n_batch': 16, 'n_sum': 16, 'n_repetitions': 32, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [1024] * 1},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [1024] * 1},
        {'flow': 'nvp', 'n_flows':  5, 'hidden_units': [1024] * 2},
        {'flow': 'nvp', 'n_flows': 10, 'hidden_units': [1024] * 2},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [1024] * 1},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [1024] * 1},
        {'flow': 'maf', 'n_flows':  5, 'hidden_units': [1024] * 2},
        {'flow': 'maf', 'n_flows': 10, 'hidden_units': [1024] * 2}
    ]

    model = RatSpn(**ratspn_kwargs)
    collect_results('mnist', 'rat-spn', model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        info = ratspn_flow_info(kwargs)
        model = RatSpnFlow(**ratspn_kwargs, **kwargs, activation='relu', regularization=1e-6)
        collect_results('mnist', info, model, LR_FLOW, data_train, data_val, data_test)
        collect_samples('mnist', info, model, N_SAMPLES, plot_fn=mnist_plot, post_fn=mnist_delogit)


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
    optimizer = tf.keras.optimizers.Adam(lr)

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


def ratspn_flow_info(kwargs):
    return 'rat-spn-' + kwargs['flow'] + str(kwargs['n_flows']) + '-d' + str(len(kwargs['hidden_units']))


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
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
