import os
import sys
import torchvision
import numpy as np
from experiments.datasets import load_unsupervised_mnist, load_supervised_mnist

from spnflow.torch.models import DgcSpn
from spnflow.torch.transforms import Dequantize, Logit, Delogit, Reshape
from experiments.utils import collect_results_generative, collect_results_discriminative


def run_experiment_mnist():
    n_classes = 10
    in_size = (1, 28, 28)

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Set the parameters for the DGC-SPN (generative setting)
    dgcspn_kwargs = [
        {'n_batch': 8, 'sum_channels':  4, 'prod_channels': 16, 'rand_state': rand_state},
        {'n_batch': 8, 'sum_channels':  8, 'prod_channels': 32, 'rand_state': rand_state},
        {'n_batch': 8, 'sum_channels': 16, 'prod_channels': 64, 'rand_state': rand_state},
    ]

    # Set the transformation (generative setting)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Reshape(*in_size),
        Dequantize(1.0 / 256.0),
        Logit(),
    ])

    # Load the dataset (generative setting)
    data_train, data_val, data_test = load_unsupervised_mnist(dataroot, transform)

    # Run the RAT-SPN experiment (generative setting)
    for kwargs in dgcspn_kwargs:
        quantiles_loc = data_train.dataset.mean_quantiles(kwargs['n_batch'])
        model = DgcSpn(in_size, quantiles_loc=quantiles_loc, **kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test)

    # Set the parameters for the DGC-SPN (discriminative setting)
    dgcspn_kwargs = [
        {'n_batch': 32, 'sum_channels': 64, 'depthwise': True, 'n_pooling': 2},
        {'n_batch': 32, 'sum_channels': 64, 'depthwise': True, 'n_pooling': 1},
        {'n_batch': 32, 'sum_channels': 64, 'depthwise': True, 'n_pooling': 0},
    ]

    # Set the transformation (discriminative setting)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.0, 1.0),
        Reshape(*in_size)
    ])

    # Load the dataset (discriminative setting)
    data_train, data_val, data_test = load_supervised_mnist(dataroot, transform)

    # Run the RAT-SPN experiment (discriminative setting)
    for kwargs in dgcspn_kwargs:
        model = DgcSpn(in_size, n_classes, dropout=0.2, **kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_discriminative('mnist', info, model, data_train, data_val, data_test)


def dgcspn_experiment_info(kwargs):
    info = 'dgcspn'
    if 'n_batch' in kwargs:
        info += '-b' + str(kwargs['n_batch'])
    if 'sum_channels' in kwargs:
        info += '-s' + str(kwargs['sum_channels'])
    if 'prod_channels' in kwargs:
        info += '-p' + str(kwargs['prod_channels'])
    if 'depthwise' in kwargs:
        info += '-d' + str(kwargs['depthwise'])
    if 'n_pooling' in kwargs:
        info += '-q' + str(kwargs['n_pooling'])
    return info


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataroot = os.environ['DATAROOT']

    dataset = sys.argv[1]
    if dataset == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
