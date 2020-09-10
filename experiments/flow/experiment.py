import sys
import torch
import numpy as np

from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.bsds300 import load_bsds300_dataset
from experiments.mnist import load_mnist_dataset
from experiments.mnist import to_images as mnist_to_images

from experiments.utils import collect_results, collect_samples
from spnflow.torch.models import RealNVP, MAF


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 128, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 128, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 128, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 128, 'activation': torch.nn.ReLU},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 128, 'activation': torch.nn.Tanh},
        {'n_flows': 10, 'depth': 1, 'units': 128, 'activation': torch.nn.Tanh},
        {'n_flows':  5, 'depth': 2, 'units': 128, 'activation': torch.nn.Tanh},
        {'n_flows': 10, 'depth': 2, 'units': 128, 'activation': torch.nn.Tanh},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_bsds300_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the MNIST dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
    ]

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test)
        collect_samples('mnist', info, model, mnist_to_images)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test)
        collect_samples('mnist', info, model, mnist_to_images)


def nvp_experiment_info(kwargs):
    return 'nvp' + '-n' + str(kwargs['n_flows']) + '-d' + str(kwargs['depth']) +\
           '-u' + str(kwargs['units']) + '-a' + kwargs['activation'].__name__


def maf_experiment_info(kwargs):
    return 'maf' + '-n' + str(kwargs['n_flows']) + '-d' + str(kwargs['depth']) +\
           '-u' + str(kwargs['units']) + '-a' + kwargs['activation'].__name__


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
