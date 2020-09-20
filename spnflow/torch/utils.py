import math
import time
import torch
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from spnflow.torch.callbacks import EarlyStopping


class RunningAverageMetric:
    """Running (batched) average metric"""
    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0.0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)


def compute_mean_quantiles(dataset, n_quantiles, n_jobs=-1):
    """
    Compute the mean quantiles of a dataset (Poon-Domingos).

    :param dataset: The dataset.
    :param n_quantiles: The number of quantiles.
    :param n_jobs: The number of jobs for dataset processing.
    :return: The mean quantiles tensor.
    """
    # Get the entire processed dataset
    n_samples = len(dataset)
    data = torch.stack(
        Parallel(n_jobs=n_jobs, batch_size=256, max_nbytes='64M')(
            delayed(dataset.__getitem__)(i) for i in range(n_samples)
        ),
        dim=0
    )

    # Split the dataset in quantiles regions
    data, indices = torch.sort(data, dim=0)
    section_quantiles = [math.floor(n_samples / n_quantiles)] * n_quantiles
    section_quantiles[-1] += n_samples % n_quantiles
    values_per_quantile = torch.split(data, section_quantiles, dim=0)

    # Compute the mean quantiles
    mean_per_quantiles = [torch.mean(x, dim=0) for x in values_per_quantile]
    return torch.stack(mean_per_quantiles, dim=0)


def torch_train(
        model,
        data_train,
        data_val,
        setting,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optimizer_class=torch.optim.Adam,
        weight_decay=0.0,
        n_workers=4,
        device=None
):
    """
    Train a Torch model.

    :param model: The model to train.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param setting: The train setting. It can be either 'generative' or 'discriminative'.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param optimizer_class: The optimizer class to use.
    :param weight_decay: L2 regularization factor.
    :param n_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :return: The train history.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train using device: ' + str(device))

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Instantiate the optimizer
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    if setting == 'generative':
        train_func = torch_train_generative
    elif setting == 'discriminative':
        train_func = torch_train_discriminative
    else:
        raise ValueError('Unknown train setting called %s' % setting)
    return train_func(model, train_loader, val_loader, optimizer, epochs, patience, device)


def torch_train_generative(model, train_loader, val_loader, optimizer, epochs, patience, device):
    """
    Train a Torch model in generative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param val_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param device: The device to use for training.
    :return: The train history.
    """
    # Instantiate the train history
    history = {
        'train': [], 'validation': []
    }

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=patience)

    # Move the model to device
    model.to(device)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training phase
        running_train_loss = RunningAverageMetric(train_loader.batch_size)
        tk = tqdm(
            train_loader, total=len(train_loader), leave=False,
            bar_format='{l_bar}{bar:32}{r_bar}', desc='Epoch %d/%d' % (epoch, epochs)
        )
        for inputs in tk:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            log_likelihoods = model(inputs)
            loss = -torch.sum(log_likelihoods)
            running_train_loss(loss.item())
            loss /= train_loader.batch_size
            loss.backward()
            optimizer.step()
            model.apply_constraints()

        # Validation phase
        running_val_loss = RunningAverageMetric(val_loader.batch_size)
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                log_likelihoods = model(inputs)
                loss = -torch.sum(log_likelihoods)
                running_val_loss(loss.item())

        # Get the average train and validation losses and print it
        end_time = time.time()
        train_loss = running_train_loss.average()
        val_loss = running_val_loss.average()
        print('Epoch %d/%d - train_loss: %.4f, validation_loss: %.4f [%ds]' %
              (epoch, epochs, train_loss, val_loss, end_time - start_time))

        # Append losses to history data
        history['train'].append(train_loss)
        history['validation'].append(val_loss)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

    return history


def torch_train_discriminative(model, train_loader, val_loader, optimizer, epochs, patience, device):
    """
    Train a Torch model in discriminative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param val_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param device: The device to use for training.
    :return: The train history.
    """
    # Instantiate the train history
    history = {
        'train': {'loss': [], 'accuracy': []},
        'validation': {'loss': [], 'accuracy': []}
    }

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=patience)

    # Move the model to device
    model.to(device)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training phase
        running_train_loss = RunningAverageMetric(train_loader.batch_size)
        running_train_hits = RunningAverageMetric(train_loader.batch_size)
        tk = tqdm(
            train_loader, total=len(train_loader), leave=False,
            bar_format='{l_bar}{bar:32}{r_bar}', desc='Epoch %d/%d' % (epoch, epochs)
        )
        for inputs, targets in tk:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = torch.nn.functional.nll_loss(outputs, targets, reduction='sum')
            running_train_loss(loss.item())
            loss /= train_loader.batch_size
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_train_hits(hits.item())

        # Validation phase
        running_val_loss = RunningAverageMetric(val_loader.batch_size)
        running_val_hits = RunningAverageMetric(val_loader.batch_size)
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = torch.log_softmax(model(inputs), dim=1)
                loss = torch.nn.functional.nll_loss(outputs, targets, reduction='sum')
                running_val_loss(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_val_hits(hits.item())

        # Get the average train and validation losses and accuracies and print it
        end_time = time.time()
        train_loss = running_train_loss.average()
        train_accuracy = running_train_hits.average()
        val_loss = running_val_loss.average()
        val_accuracy = running_val_hits.average()
        print('Epoch %d/%d - train_loss: %.4f, validation_loss: %.4f, train_acc: %.1f%%, validation_acc: %.1f%% [%ds]' %
              (epoch, epochs, train_loss, val_loss, train_accuracy * 100, val_accuracy * 100, end_time - start_time))

        # Append losses and accuracies to history data
        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_accuracy)
        history['validation']['loss'].append(val_loss)
        history['validation']['accuracy'].append(val_accuracy)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

    return history


def torch_test(
        model,
        data_test,
        setting,
        batch_size=100,
        n_workers=4,
        device=None
):
    """
    Test a Torch model.

    :param model: The model to test.
    :param data_test: The test dataset.
    :param setting: The test setting. It can be either 'generative' or 'discriminative'.
    :param batch_size: The batch size for testing.
    :param n_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :return: The mean log-likelihood and two standard deviations if setting='generative'.
             The negative log-likelihood and accuracy if setting='discriminative'.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Test the model
    if setting == 'generative':
        test_func = torch_test_generative
    elif setting == 'discriminative':
        test_func = torch_test_discriminative
    else:
        raise ValueError('Unknown test setting called %s' % setting)
    return test_func(model, test_loader, device)


def torch_test_generative(model, test_loader, device):
    """
    Test a Torch model in generative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :return: The mean log-likelihood and two standard deviations.
    """
    test_ll = np.array([])
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            ll = model(inputs).cpu().numpy().flatten()
            test_ll = np.hstack((test_ll, ll))
    mu_ll = np.mean(test_ll)
    sigma_ll = 2.0 * np.std(test_ll) / np.sqrt(len(test_ll))
    return mu_ll, sigma_ll


def torch_test_discriminative(model, test_loader, device):
    """
    Test a Torch model in discriminative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :return: The negative log-likelihood and accuracy.
    """
    running_loss = RunningAverageMetric(test_loader.batch_size)
    running_hits = RunningAverageMetric(test_loader.batch_size)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = torch.nn.functional.nll_loss(outputs, targets, reduction='sum')
            running_loss(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            hits = torch.eq(predictions, targets).sum()
            running_hits(hits.item())
    return running_loss.average(), running_hits.average()
