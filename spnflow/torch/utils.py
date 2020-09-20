import math
import time
import torch
import numpy as np
from joblib import Parallel, delayed
from spnflow.torch.callbacks import EarlyStopping


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


def torch_train_generative(
        model,
        data_train,
        data_val,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optim=torch.optim.Adam,
        device=None,
        num_workers=2,
):
    """
    Train a Torch model by maximizing the log-likelihood.

    :param model: The model to train.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param patience: The number of consecutive epochs to wait until no improvements of the validation loss occurs.
    :param optim: The optimizer to use.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param num_workers: The number of workers for data loading.
    """
    # Instantiate the train history
    history = {
        'train': [], 'validation': []
    }

    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train using device: ' + str(device))

    # Print the model
    print(model)

    # Move the model to the device
    model.to(device)

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Instantiate the optimizer
    optimizer = optim(model.parameters(), lr=lr)

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=patience)

    # Train the model
    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        train_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            log_likelihoods = model(inputs)
            loss = -log_likelihoods.sum()
            train_loss += loss.item()
            loss /= batch_size
            loss.backward()
            optimizer.step()
            model.apply_constraints()
        train_loss /= len(train_loader) * batch_size

        # Compute the validation loss
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                log_likelihoods = model(inputs)
                val_loss += -log_likelihoods.sum().item()
            val_loss /= len(val_loader) * batch_size

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

    return history


def torch_train_discriminative(
        model,
        data_train,
        data_val,
        lr=1e-3,
        batch_size=100,
        epochs=250,
        patience=10,
        optim=torch.optim.Adam,
        device=None,
        num_workers=2,
):
    """
    Train a Torch model by minimizing the categorical cross entropy.

    :param model: The model to train.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param patience: The number of consecutive epochs to wait until no improvements of the validation loss occurs.
    :param optim: The optimizer to use.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param num_workers: The number of workers for data loading.
    """
    # Instantiate the train history
    history = {
        'train': {'loss': [], 'accuracy': []},
        'validation': {'loss': [], 'accuracy': []}
    }

    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train using device: ' + str(device))

    # Print the model
    print(model)

    # Move the model to the device
    model.to(device)

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Instantiate the optimizer
    optimizer = optim(model.parameters(), lr=lr)

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=patience)

    # Train the model
    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        train_loss = 0.0
        train_hits = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = torch.nn.functional.nll_loss(outputs, targets, reduction='sum')
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_hits += torch.eq(preds, targets).sum().item()
            loss /= batch_size
            loss.backward()
            optimizer.step()
            model.apply_constraints()
        train_loss /= len(train_loader) * batch_size
        train_hits /= len(train_loader) * batch_size

        # Compute the validation loss
        val_loss = 0.0
        val_hits = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = torch.log_softmax(model(inputs), dim=1)
                val_loss += torch.nn.functional.nll_loss(outputs, targets, reduction='sum').item()
                preds = torch.argmax(outputs, dim=1)
                val_hits += torch.eq(preds, targets).sum().item()
            val_loss /= len(val_loader) * batch_size
            val_hits /= len(val_loader) * batch_size

        end_time = time.time()
        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_hits)
        history['validation']['loss'].append(val_loss)
        history['validation']['accuracy'].append(val_hits)
        elapsed_time = end_time - start_time
        print('[%4d] train_loss: %.4f, val_loss: %.4f, train_acc %.1f, val_acc: %.1f - %ds' %
              (epoch + 1, train_loss, val_loss, train_hits * 100, val_hits * 100, elapsed_time))

        # Check if training should stop
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

    return history


def torch_test_generative(
        model,
        data_test,
        batch_size=100,
        device=None,
        num_workers=2,
):
    """
    Test a Torch model by its log-likelihood.

    :param model: The model to test.
    :param data_test: The test dataset.
    :param batch_size: The batch size for testing.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param num_workers: The number of workers for data loading.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Test the model
    test_ll = np.array([])
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            ll = model(inputs).cpu().numpy().flatten()
            test_ll = np.hstack((test_ll, ll))

    mu_ll = np.mean(test_ll)
    sigma_ll = 2.0 * np.std(test_ll) / np.sqrt(len(test_ll))
    return mu_ll, sigma_ll


def torch_test_discriminative(
        model,
        dataset,
        batch_size=100,
        device=None,
        num_workers=2,
):
    """
    Test a Torch model by its log-likelihood.

    :param model: The model to test.
    :param dataset: The test dataset.
    :param batch_size: The batch size for testing.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param num_workers: The number of workers for data loading.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Test the model
    test_loss = 0.0
    test_hits = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            test_loss += torch.nn.functional.nll_loss(outputs, targets, reduction='sum').item()
            preds = torch.argmax(outputs, dim=1)
            test_hits += torch.eq(preds, targets).sum().item()
        test_loss /= len(test_loader) * batch_size
        test_hits /= len(test_loader) * batch_size

    return test_loss, test_hits
