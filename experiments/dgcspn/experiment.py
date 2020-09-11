import os
import sys
import torchvision
import numpy as np
from experiments.datasets import load_unsupervised_mnist, load_supervised_mnist

from spnflow.torch.models import DgcSpn
from spnflow.torch.transforms import Dequantize, Normalize, Logit, Delogit, Reshape
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_samples


def run_experiment_mnist():
    n_classes = 10
    image_size = (1, 28, 28)

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Set the parameters for the DGC-SPN
    dgcspn_kwargs = [
        {
            'in_size': image_size, 'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 2,
            'rand_state': rand_state,
        },
        {
            'in_size': image_size, 'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 1,
            'rand_state': rand_state
        },
        {
            'in_size': image_size, 'n_batch': 16, 'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 0,
            'rand_state': rand_state
        },
    ]

    # Set the transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Reshape(*image_size),
        Dequantize(),
        Normalize(255.0),
        Logit(),
    ])

    # Set the tensor sample to image transformation
    sample_transform = torchvision.transforms.Compose([
        Delogit(),
        Reshape(*image_size)
    ])

    # Load the dataset (generative setting)
    data_train, data_val, data_test = load_unsupervised_mnist(dataroot, transform)
    init_args = {'dataset': data_train}

    # Run the RAT-SPN experiment (generative setting)
    for kwargs in dgcspn_kwargs:
        model = DgcSpn(**kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_generative('mnist', info, model, data_train, data_val, data_test, init_args=init_args)
        collect_samples('mnist', info, model, n_samples=(5, 5), transform=sample_transform)

    # Set the transformation (discriminative setting)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.0, 1.0),
        Reshape(*image_size)
    ])

    # Load the dataset (discriminative setting)
    data_train, data_val, data_test = load_supervised_mnist(dataroot, transform)
    init_args = {'dataset': data_train}

    # Run the RAT-SPN experiment (discriminative setting)
    for kwargs in dgcspn_kwargs:
        model = DgcSpn(out_classes=n_classes, **kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results_discriminative('mnist', info, model, data_train, data_val, data_test, init_args=init_args)


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
