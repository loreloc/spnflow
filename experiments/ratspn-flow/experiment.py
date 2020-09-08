import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.bsds300 import load_bsds300_dataset
from experiments.mnist import load_mnist_dataset
from experiments.mnist import plot_sample as mnist_plot_sample
from spnflow.torch.models import RealNVP, MAF, RatSpn, RatSpnFlow
from spnflow.torch.utils import torch_train_generative, torch_test_generative

EPOCHS = 1000
BATCH_SIZE = 100
PATIENCE = 30
LR_FLOW = 1e-3
LR_RAT = 5e-4
LR_RAT_FLOW = 1e-4
N_SAMPLES = 100
N_IMG_SAMPLES = (8, 8)


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 1, 'rg_repetitions': 8, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 128, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'units': 128, 'activation': torch.nn.ReLU},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('power', info, model, data_train, data_val, data_test, LR_RAT)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test, LR_FLOW)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test, LR_FLOW)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test, LR_RAT_FLOW)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('power', info, model, data_train, data_val, data_test, LR_RAT_FLOW)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 1, 'rg_repetitions': 8, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 128, 'activation': torch.nn.Tanh},
        {'n_flows': 10, 'units': 128, 'activation': torch.nn.Tanh},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('gas', info, model, data_train, data_val, data_test, LR_RAT)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test, LR_FLOW)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test, LR_FLOW)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test, LR_RAT_FLOW)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('gas', info, model, data_train, data_val, data_test, LR_RAT_FLOW)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('hepmass', info, model, data_train, data_val, data_test, LR_RAT)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test, LR_FLOW)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test, LR_FLOW)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test, LR_RAT_FLOW)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('hepmass', info, model, data_train, data_val, data_test, LR_RAT_FLOW)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('miniboone', info, model, data_train, data_val, data_test, LR_RAT)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test, LR_FLOW)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test, LR_FLOW)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test, LR_RAT_FLOW)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('miniboone', info, model, data_train, data_val, data_test, LR_RAT_FLOW)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_bsds300_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 2, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 512, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('bsds300', info, model, data_train, data_val, data_test, LR_RAT)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test, LR_FLOW)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test, LR_FLOW)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test, LR_RAT_FLOW)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('bsds300', info, model, data_train, data_val, data_test, LR_RAT_FLOW)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the MNIST dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'rg_depth': 3, 'rg_repetitions': 16, 'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'n_flows':  5, 'units': 1024, 'activation': torch.nn.ReLU},
        {'n_flows': 10, 'units': 1024, 'activation': torch.nn.ReLU},
    ]

    # RAT-SPN experiment
    model = RatSpn(n_features, **ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('mnist', info, model, data_train, data_val, data_test, LR_RAT)
    collect_samples('mnist', info, model, mnist_plot_sample)

    # RealNVP experiments
    for kwargs in flow_kwargs:
        model = RealNVP(n_features, **kwargs)
        info = nvp_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR_FLOW)
        collect_samples('mnist', info, model, mnist_plot_sample)

    # MAF experiments
    for kwargs in flow_kwargs:
        model = MAF(n_features, **kwargs)
        info = maf_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR_FLOW)
        collect_samples('mnist', info, model, mnist_plot_sample)

    # RAT-SPN + RealNVP experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='nvp', **kwargs)
        info = ratspn_nvp_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR_RAT_FLOW)
        collect_samples('mnist', info, model, mnist_plot_sample)

    # RAT-SPN + MAF experiments
    for kwargs in flow_kwargs:
        kwargs = {**ratspn_kwargs, **kwargs}
        model = RatSpnFlow(n_features, flow='maf', **kwargs)
        info = ratspn_maf_experiment_info(kwargs)
        collect_results('mnist', info, model, data_train, data_val, data_test, LR_RAT_FLOW)
        collect_samples('mnist', info, model, mnist_plot_sample)


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


def collect_samples(dataset, info, model, plot_func=None, **plot_kwargs):
    # Initialize the results filepath
    filepath = os.path.join('samples', dataset + '_' + info)

    # If image_func is specified, save the samples as images, otherwise save them in CSV format
    if not plot_func:
        samples = model.sample(N_SAMPLES).cpu().numpy()
        pd.DataFrame(samples).to_csv(filepath + '.csv')
    else:
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


def nvp_experiment_info(kwargs):
    return 'nvp' + '-n' + str(kwargs['n_flows']) + '-u' + str(kwargs['units']) + '-a' + kwargs['activation'].__name__


def maf_experiment_info(kwargs):
    return 'maf' + '-n' + str(kwargs['n_flows']) + '-u' + str(kwargs['units']) + '-a' + kwargs['activation'].__name__


def ratspn_experiment_info(kwargs):
    return 'ratspn' + '-d' + str(kwargs['rg_depth']) + '-r' + str(kwargs['rg_repetitions']) +\
           '-b' + str(kwargs['n_batch']) + '-s' + str(kwargs['n_sum'])


def ratspn_nvp_experiment_info(kwargs):
    return ratspn_experiment_info(kwargs) + '-' + nvp_experiment_info(kwargs)


def ratspn_maf_experiment_info(kwargs):
    return ratspn_experiment_info(kwargs) + '-' + maf_experiment_info(kwargs)


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
