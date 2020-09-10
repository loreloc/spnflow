import sys
import numpy as np
from experiments.mnist import load_mnist_dataset
from experiments.mnist import to_images as mnist_to_images

from experiments.utils import collect_results, collect_samples
from spnflow.torch.models import DgcSpn

LR = 1e-3


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state, flatten=False)
    in_size = data_train.shape[1:]

    # Set the parameters for the DGC-SPN
    dgcspn_kwargs = [
        {
            'in_size': in_size, 'n_batch': 16,
            'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 2,
            'rand_state': rand_state
        },
        {
            'in_size': in_size, 'n_batch': 16,
            'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 1,
            'rand_state': rand_state
        },
        {
            'in_size': in_size, 'n_batch': 16,
            'prod_channels': 32, 'sum_channels': 64, 'n_pooling': 0,
            'rand_state': rand_state
        },
    ]

    for kwargs in dgcspn_kwargs:
        model = DgcSpn(**kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR, init_args={'data': data_train})
        collect_samples('mnist', info, model, mnist_to_images)


def dgcspn_experiment_info(kwargs):
    return 'dgcspn-b' + str(kwargs['n_batch']) +\
           '-p' + str(kwargs['prod_channels']) + '-s' + str(kwargs['sum_channels'])


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataset = sys.argv[1]
    if dataset == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
