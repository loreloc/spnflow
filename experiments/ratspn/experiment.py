import os
import sys
import numpy as np
import torchvision

from experiments.datasets import load_dataset, load_unsupervised_mnist, load_supervised_mnist

from spnflow.torch.models import RatSpn
from spnflow.torch.transforms import Flatten, Dequantize, Logit, Delogit, Reshape
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_samples


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'POWER', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 1, 'rg_repetitions': 8, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 1, 'rg_repetitions': 8, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # RAT-SPN experiment
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('power', info, model, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'GAS', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 1, 'rg_repetitions': 8, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 1, 'rg_repetitions': 8, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # RAT-SPN experiment
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('gas', info, model, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'HEPMASS', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # RAT-SPN experiment
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('hepmass', info, model, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'MINIBOONE', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # RAT-SPN experiment
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('miniboone', info, model, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'BSDS300', rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # RAT-SPN experiment
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('bsds300', info, model, data_train, data_val, data_test)


def run_experiment_mnist():
    n_features = 784
    n_classes = 10
    image_size = (1, 28, 28)

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = [
        {'rg_depth': 3, 'rg_repetitions': 16, 'n_batch':  8, 'n_sum':  8, 'rand_state': rand_state},
        {'rg_depth': 3, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
        {'rg_depth': 4, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
        {'rg_depth': 4, 'rg_repetitions': 32, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state},
    ]

    # Set the transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Flatten(),
        Dequantize(1.0 / 256.0),
        Logit()
    ])

    # Set the tensor sample to image transformation
    sample_transform = torchvision.transforms.Compose([
        Delogit(),
        Reshape(*image_size)
    ])

    # Load the dataset (generative setting)
    data_train, data_val, data_test = load_unsupervised_mnist(dataroot, transform)

    # Run the RAT-SPN experiment (generative setting)
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test)
        collect_samples('mnist', info, model, n_samples=(8, 8), transform=sample_transform)

    # Set the transformation (discriminative setting)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.1307, 0.3081),
        Flatten()
    ])

    # Load the dataset (discriminative setting)
    data_train, data_val, data_test = load_supervised_mnist(dataroot, transform)

    # Run the RAT-SPN experiment (discriminative setting)
    for kwargs in ratspn_kwargs:
        model = RatSpn(n_features, n_classes, dropout=0.2, **kwargs)
        info = ratspn_experiment_info(kwargs)
        collect_results_discriminative('mnist', info, model, data_train, data_val, data_test)


def ratspn_experiment_info(kwargs):
    info = 'ratspn'
    if 'rg_depth' in kwargs:
        info += '-d' + str(kwargs['rg_depth'])
    if 'rg_repetitions' in kwargs:
        info += '-r' + str(kwargs['rg_repetitions'])
    if 'n_batch' in kwargs:
        info += '-b' + str(kwargs['n_batch'])
    if 'n_sum' in kwargs:
        info += '-s' + str(kwargs['n_sum'])
    return info


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
    elif dataset == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
