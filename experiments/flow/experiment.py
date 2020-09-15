import os
import sys
import torch
import torchvision
import numpy as np

from experiments.datasets import load_dataset, load_unsupervised_mnist

from spnflow.torch.models import RealNVP, MAF
from spnflow.torch.transforms import Flatten, Dequantize, Logit, Delogit, Reshape
from experiments.utils import collect_results_generative, collect_samples


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'POWER', rand_state)
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
        collect_results_generative('power', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('power', info, model, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'GAS', rand_state)
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
        collect_results_generative('gas', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('gas', info, model, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'HEPMASS', rand_state)
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
        collect_results_generative('hepmass', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('hepmass', info, model, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'MINIBOONE', rand_state)
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
        collect_results_generative('miniboone', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('miniboone', info, model, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_dataset(dataroot, 'BSDS300', rand_state)
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
        collect_results_generative('bsds300', info, model, data_train, data_val, data_test)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('bsds300', info, model, data_train, data_val, data_test)


def run_experiment_mnist():
    n_features = 784
    image_size = (1, 28, 28)

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows':  5, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
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

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test)
        collect_samples('mnist', info, model, n_samples=(8, 8), transform=sample_transform)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test)
        collect_samples('mnist', info, model, n_samples=(8, 8), transform=sample_transform)


def nvp_experiment_info(kwargs):
    info = 'nvp'
    if 'n_flows' in kwargs:
        info += '-n' + str(kwargs['n_flows'])
    if 'depth' in kwargs:
        info += '-d' + str(kwargs['depth'])
    if 'units' in kwargs:
        info += '-u' + str(kwargs['units'])
    if 'activation' in kwargs:
        info += '-a' + kwargs['activation'].__name__
    return info


def maf_experiment_info(kwargs):
    info = 'maf'
    if 'n_flows' in kwargs:
        info += '-n' + str(kwargs['n_flows'])
    if 'depth' in kwargs:
        info += '-d' + str(kwargs['depth'])
    if 'units' in kwargs:
        info += '-u' + str(kwargs['units'])
    if 'activation' in kwargs:
        info += '-a' + kwargs['activation'].__name__
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
