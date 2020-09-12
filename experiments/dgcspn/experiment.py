import os
import sys
import torchvision
import numpy as np
from experiments.datasets import load_unsupervised_mnist, load_supervised_mnist

from spnflow.torch.models import DgcSpn
from spnflow.torch.transforms import Dequantize, Normalize, Logit, Delogit, Reshape
from experiments.utils import collect_results_generative, collect_results_discriminative


def run_experiment_mnist():
    n_classes = 10
    in_size = (1, 28, 28)

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Set the parameters for the DGC-SPN
    dgcspn_kwargs = [
        {'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 2, 'rand_state': rand_state},
        {'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 1, 'rand_state': rand_state},
        {'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 0, 'rand_state': rand_state},
    ]
    # Set the transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Reshape(*in_size),
        Dequantize(),
        Normalize(255.0),
        Logit(),
    ])

    # Load the dataset (generative setting)
    data_train, data_val, data_test = load_unsupervised_mnist(dataroot, transform)

    # Run the RAT-SPN experiment (generative setting)
    for kwargs in dgcspn_kwargs:
        quantiles = data_train.dataset.mean_quantiles(kwargs['n_batch'])
        model = DgcSpn(in_size, quantiles_loc=True, quantiles=quantiles, **kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test)

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
        model = DgcSpn(in_size, n_classes, **kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_discriminative('mnist', info, model, data_train, data_val, data_test)


def dgcspn_experiment_info(kwargs):
    return 'dgcspn-b' + str(kwargs['n_batch']) + '-p' + str(kwargs['prod_channels']) +\
           '-s' + str(kwargs['sum_channels']) + '-q' + str(kwargs['n_pooling'])


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataroot = os.environ['DATAROOT']

    dataset = sys.argv[1]
    if dataset == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
