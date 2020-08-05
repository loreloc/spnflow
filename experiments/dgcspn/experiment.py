import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.mnist import load_mnist_dataset
from spnflow.torch.models import SpatialSpn
from spnflow.torch.utils import torch_train_generative, torch_test_generative

EPOCHS = 1000
BATCH_SIZE = 100
PATIENCE = 30
LR = 1e-6


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state, logit_space=False, flatten=False)
    in_size = data_train.shape[1:]

    # Set the parameters for the DGC-SPN
    dgcspn_kwargs = [
        {'in_size': in_size, 'n_batch': 16, 'prod_channels': 32, 'sum_channels': 16, 'rand_state': rand_state},
    ]

    for kwargs in dgcspn_kwargs:
        model = SpatialSpn(**kwargs)
        info = dgcspn_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR)


def collect_results(dataset, info, model, data_train, data_val, data_test, lr):
    # Train the model
    history = torch_train_generative(model, data_train, data_val, torch.optim.Adam, lr, BATCH_SIZE, PATIENCE, EPOCHS)

    # Test the model
    (mu_ll, sigma_ll) = torch_test_generative(model, data_test, BATCH_SIZE)

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Mean Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two StdDev. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')

    # Plot the training history
    filepath = os.path.join('histories', dataset + '_' + info + '.png')
    plt.xlim(0, EPOCHS)
    plt.plot(history['train'])
    plt.plot(history['validation'])
    plt.title('Log-Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(filepath)
    plt.clf()


def dgcspn_experiment_info(kwargs):
    return 'dgcspn'


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage:\n\tpython experiment.py <dataset>")

    dataset = sys.argv[1]
    if dataset == 'mnist':
        run_experiment_mnist()
    else:
        raise NotImplementedError("Unknown dataset: " + dataset)
