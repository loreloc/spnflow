import os
import sys
import numpy as np

from experiments.datasets import load_dataset

from spnflow.structure.leaf import Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.inference import log_likelihood


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'POWER', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the SPN
    spn_kwargs = {
        'distributions': [Gaussian] * n_features,
        'learn_leaf': 'mle',
        'split_rows': 'kmeans',
        'split_cols': 'rdc',
    }

    # Learn the density estimator structure and parameters
    for mrs in [128, 256, 512]:
        spn = learn_estimator(data_train, **spn_kwargs, min_rows_slice=mrs)
        collect_results('power', 'spn-' + str(mrs), spn, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'GAS', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the SPN
    spn_kwargs = {
        'distributions': [Gaussian] * n_features,
        'learn_leaf': 'mle',
        'split_rows': 'kmeans',
        'split_cols': 'rdc',
    }

    # Learn the density estimator structure and parameters
    for mrs in [128, 256, 512]:
        spn = learn_estimator(data_train, **spn_kwargs, min_rows_slice=mrs)
        collect_results('gas', 'spn-' + str(mrs), spn, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'HEPMASS', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the SPN
    spn_kwargs = {
        'distributions': [Gaussian] * n_features,
        'learn_leaf': 'mle',
        'split_rows': 'kmeans',
        'split_cols': 'rdc',
    }

    # Learn the density estimator structure and parameters
    for mrs in [64, 128, 256]:
        spn = learn_estimator(data_train, **spn_kwargs, min_rows_slice=mrs)
        collect_results('hepmass', 'spn-' + str(mrs), spn, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'MINIBOONE', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the SPN
    spn_kwargs = {
        'distributions': [Gaussian] * n_features,
        'learn_leaf': 'mle',
        'split_rows': 'kmeans',
        'split_cols': 'rdc',
    }

    # Learn the density estimator structure and parameters
    for mrs in [64, 128, 256]:
        spn = learn_estimator(data_train, **spn_kwargs, min_rows_slice=mrs)
        collect_results('miniboone', 'spn-' + str(mrs), spn, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'BSDS300', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the SPN
    spn_kwargs = {
        'distributions': [Gaussian] * n_features,
        'learn_leaf': 'mle',
        'split_rows': 'kmeans',
        'split_cols': 'rdc',
    }

    # Learn the density estimator structure and parameters
    for mrs in [256, 512]:
        spn = learn_estimator(data_train, **spn_kwargs, min_rows_slice=mrs)
        collect_results('bsds300', 'spn-' + str(mrs), spn, data_test)


def collect_results(dataset, info, spn, data_test):
    # Compute the log-likelihoods for the test set
    ll = log_likelihood(spn, data_test)
    mu_ll = np.mean(ll)
    sigma_ll = np.std(ll) / np.sqrt(data_test.shape[0])

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Avg. Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two Std. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataroot = os.environ['DATAROOT']

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
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
