import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments.power import load_power_dataset
from experiments.gas import load_gas_dataset
from experiments.hepmass import load_hepmass_dataset
from experiments.miniboone import load_miniboone_dataset
from experiments.bsds300 import load_bsds300_dataset
from experiments.mnist import load_mnist_dataset

from spnflow.torch.models import RatSpn, RatSpnFlow
from spnflow.torch.callbacks import EarlyStopping
from spnflow.torch.constraints import ScaleClipper

EPOCHS = 1000
BATCH_SIZE = 100
PATIENCE = 30
LR_RAT = 5e-4
LR_FLOW = 1e-4


def run_experiment_power():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the power dataset
    data_train, data_val, data_test = load_power_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 1, 'rg_repetitions': 8,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 128, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 128, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 128, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 128, 'activation': torch.nn.ReLU},
    ]

    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('power', info, model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('power', info, model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_gas():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the gas dataset
    data_train, data_val, data_test = load_gas_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 1, 'rg_repetitions': 8,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 128, 'activation': torch.nn.Tanh},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 128, 'activation': torch.nn.Tanh},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 128, 'activation': torch.nn.Tanh},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 128, 'activation': torch.nn.Tanh},
    ]

    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('gas', info, model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('gas', info, model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_hepmass():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the hepmass dataset
    data_train, data_val, data_test = load_hepmass_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 2, 'rg_repetitions': 16,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('hepmass', info, model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('hepmass', info, model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_miniboone():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the miniboone dataset
    data_train, data_val, data_test = load_miniboone_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 2, 'rg_repetitions': 16,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('miniboone', info, model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('miniboone', info, model, LR_FLOW, data_train, data_val, data_test)


def run_experiment_bsds300():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the BSDS300 dataset
    data_train, data_val, data_test = load_bsds300_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 2, 'rg_repetitions': 16,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 512, 'activation': torch.nn.ReLU},
    ]

    lr_rat = LR_RAT * 1e-1
    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('bsds300', info, model, lr_rat, data_train, data_val, data_test)

    lr_flow = LR_FLOW * 1e-1
    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('bsds300', info, model, lr_flow, data_train, data_val, data_test)


def run_experiment_mnist():
    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the mnist dataset
    data_train, data_val, data_test = load_mnist_dataset(rand_state)
    _, n_features = data_train.shape

    # Set the parameters for the RAT-SPNs
    ratspn_kwargs = {
        'in_features': n_features, 'rg_depth': 3, 'rg_repetitions': 32,
        'n_batch': 16, 'n_sum': 16, 'rand_state': rand_state
    }

    # Set the parameters for the normalizing flows conditioners
    flow_kwargs = [
        {'flow': 'nvp', 'n_flows':  5, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 1, 'units': 1024, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows':  5, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
        {'flow': 'nvp', 'n_flows': 10, 'depth': 2, 'units': 1024, 'activation': torch.nn.ReLU},
    ]

    model = RatSpn(**ratspn_kwargs)
    info = ratspn_experiment_info(ratspn_kwargs)
    collect_results('mnist', info, model, LR_RAT, data_train, data_val, data_test)

    for kwargs in flow_kwargs:
        model = RatSpnFlow(**ratspn_kwargs, **kwargs)
        info = ratspn_flow_experiment_info(kwargs)
        collect_results('mnist', info, model, LR_FLOW, data_train, data_val, data_test)


def collect_results(dataset, info, model, lr, data_train, data_val, data_test):
    # Run the experiment and get the results
    history, (mu_ll, sigma_ll) = experiment_log_likelihood(model, lr, data_train, data_val, data_test)

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


def experiment_log_likelihood(model, lr, data_train, data_val, data_test):
    # Get the device to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))

    # Print the model
    print(model)

    # Move the model to the device
    model.to(device)

    # Instantiate the train history
    history = {
        'train': [], 'validation': []
    }

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Instantiate the scale constraint
    constraint = ScaleClipper()
    constraint.to(device)

    # Train the model
    for epoch in range(EPOCHS):
        start_time = time.time()

        # Training phase
        train_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            log_likelihoods = model(inputs)
            loss = -torch.mean(log_likelihoods)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            constraint(model.constrained_module)

        train_loss /= len(train_loader)

        # Compute the validation loss
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                log_likelihoods = model(inputs)
                loss = -torch.mean(log_likelihoods)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        end_time = time.time()

        history['train'].append(train_loss)
        history['validation'].append(val_loss)
        elapsed_time = end_time - start_time
        print('[%4d] train_loss: %.4f, validation_loss: %.4f - %ds' % (epoch + 1, train_loss, val_loss, elapsed_time))

        # Check if training should stop
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

    # Test the model
    test_ll = np.array([])
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            ll = model(inputs)
            mean_ll = torch.mean(ll).cpu().numpy()
            test_ll = np.hstack((test_ll, mean_ll))

    mu_ll = np.mean(test_ll)
    sigma_ll = np.std(test_ll) / np.sqrt(len(test_ll))
    return history, (mu_ll, sigma_ll)


def ratspn_experiment_info(kwargs):
    return 'rat-spn-d' + str(kwargs['rg_depth']) + '-r' + str(kwargs['rg_repetitions']) +\
           '-b' + str(kwargs['n_batch']) + '-s' + str(kwargs['n_sum'])


def ratspn_flow_experiment_info(kwargs):
    return 'rat-spn-' + kwargs['flow'] + '-n' + str(kwargs['n_flows']) +\
           '-d' + str(kwargs['depth']) + '-u' + str(kwargs['units'])


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
