import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.mnist import load_mnist_dataset
from experiments.mnist import plot_sample as mnist_plot_sample
from spnflow.torch.models import DgcSpn
from spnflow.torch.utils import torch_train_generative, torch_test_generative

EPOCHS = 1000
BATCH_SIZE = 100
PATIENCE = 30
LR = 1e-3
N_IMG_SAMPLES = (8, 8)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state, flatten=False)
    in_size = data_train.shape[1:]

    # Set the parameters for the DGC-SPN
    dgcspn_kwargs = [
        {
            'in_size': in_size, 'n_batch': 8,
            'prod_channels': 16, 'sum_channels': 8, 'n_pooling': 2,
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
        collect_results('mnist', info, model, data_train, data_val, data_test, LR)
        collect_samples('mnist', info, model, mnist_plot_sample)


def collect_results(dataset, info, model, data_train, data_val, data_test, lr):
    # Train the model
    history = torch_train_generative(
        model, data_train, data_val,
        torch.optim.Adam, lr, BATCH_SIZE, PATIENCE, EPOCHS,
        init_args={'data': data_train}
    )

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


def collect_samples(dataset, info, model, plot_func, **plot_kwargs):
    # Initialize the results filepath
    filepath = os.path.join('samples', dataset + '_' + info)

    # Save the samples as images
    rows, cols = N_IMG_SAMPLES
    samples = model.sample(rows * cols).cpu().numpy()
    n_samples, n_features = samples.shape
    fig, axs = plt.subplots(rows, cols, figsize=(rows, cols))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    axs = axs.flatten()
    for ax, image in zip(axs, samples):
        ax.set_axis_off()
        plot_func(ax, image, **plot_kwargs)
    dpi = int(np.sqrt(n_features))
    plt.savefig(filepath + '.png', dpi=dpi)
    plt.clf()


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
