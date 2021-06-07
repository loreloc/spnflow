import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from deeprob.torch.utils import get_optimizer_class
from deeprob.torch.callbacks import EarlyStopping
from deeprob.torch.metrics import RunningAverageMetric
from deeprob.torch.models.flows import AbstractNormalizingFlow


def train_model(
        model,
        data_train,
        data_val,
        setting,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optimizer='adam',
        optimizer_kwargs=None,
        weight_decay=0.0,
        train_base=True,
        class_weights=None,
        num_workers=0,
        device=None,
        verbose=True
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
    :param optimizer: The optimizer to use.
    :param optimizer_kwargs: A dictionary containing additional optimizer parameters.
    :param weight_decay: L2 regularization factor.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param class_weights: The class weights (for im-balanced datasets). Used only if setting='discriminative'.
    :param num_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :return: The train history.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Train using device: {}'.format(device))

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Move the model to device
    model.to(device)

    # Instantiate the optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = get_optimizer_class(optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay, **optimizer_kwargs
    )

    # Train the model
    if setting == 'generative':
        return train_generative(
            model, train_loader, val_loader,
            optimizer, epochs, patience, train_base, device, verbose
        )
    elif setting == 'discriminative':
        return train_discriminative(
            model, train_loader, val_loader,
            optimizer, epochs, patience, train_base, class_weights, device, verbose
        )
    else:
        raise ValueError('Unknown train setting called {}'.format(setting))


def train_generative(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        patience,
        train_base,
        device,
        verbose
):
    """
    Train a Torch model in generative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param val_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param device: The device to use for training.
    :param verbose: Whether to enable verbose mode.
    :return: The train history.
    """
    # Instantiate the train history
    history = {
        'train': [], 'validation': []
    }

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(model, patience=patience)

    # Instantiate the running average metrics
    running_train_loss = RunningAverageMetric(train_loader.batch_size)
    running_val_loss = RunningAverageMetric(val_loader.batch_size)

    for epoch in range(epochs):
        start_time = time.perf_counter()

        # Reset the metrics
        running_train_loss.reset()
        running_val_loss.reset()

        # Initialize the tqdm train data loader, if verbose is specified
        if verbose:
            tk_train = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Train Epoch {}/{}'.format(epoch + 1, epochs)
            )
        else:
            tk_train = train_loader

        # Make sure the model is set to train mode
        if isinstance(model, AbstractNormalizingFlow):
            model.train(base_mode=train_base)
        else:
            model.train()

        # Training phase
        for inputs in tk_train:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            loss = -torch.mean(model(inputs))
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            running_train_loss(loss.item() * train_loader.batch_size)

        # Initialize the tqdm validation data loader, if verbose is specified
        if verbose:
            tk_val = tqdm(
                val_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Validation Epoch {}/{}'.format(epoch + 1, epochs)
            )
        else:
            tk_val = val_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        with torch.no_grad():
            for inputs in tk_val:
                inputs = inputs.to(device)
                loss = -torch.mean(model(inputs))
                running_val_loss(loss.item() * val_loader.batch_size)

        # Get the average train and validation losses and print it
        end_time = time.perf_counter()
        train_loss = running_train_loss.average()
        val_loss = running_val_loss.average()
        print('Epoch {}/{} - train_loss: {:.4f}, validation_loss: {:.4f} [{}s]'.format(
            epoch + 1, epochs, train_loss, val_loss, int(end_time - start_time)
        ))

        # Append losses to history data
        history['train'].append(train_loss)
        history['validation'].append(val_loss)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: {:.4f}'.format(early_stopping.best_loss))
            break

    # Load the best parameters state according to early stopping
    model.load_state_dict(early_stopping.best_state)
    return history


def train_discriminative(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        patience,
        train_base,
        class_weights,
        device,
        verbose
):
    """
    Train a Torch model in discriminative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param val_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param train_base: Whether to train the input base module. Only applicable for normalizing flows.
    :param class_weights: The class weights (for im-balanced datasets).
    :param device: The device to use for training.
    :param verbose: Whether to enable verbose mode.
    :return: The train history.
    """
    # Instantiate the train history
    history = {
        'train': {'loss': [], 'accuracy': []},
        'validation': {'loss': [], 'accuracy': []}
    }

    # Instantiate the loss
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weight = None
    nll_loss = torch.nn.NLLLoss(weight=weight)

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(model, patience=patience)

    # Instantiate the running average metrics
    running_train_loss = RunningAverageMetric(train_loader.batch_size)
    running_train_hits = RunningAverageMetric(train_loader.batch_size)
    running_val_loss = RunningAverageMetric(val_loader.batch_size)
    running_val_hits = RunningAverageMetric(val_loader.batch_size)

    for epoch in range(epochs):
        start_time = time.perf_counter()

        # Reset the metrics
        running_train_loss.reset()
        running_train_hits.reset()
        running_val_loss.reset()
        running_val_hits.reset()

        # Initialize the tqdm train data loader, if verbose is enabled
        if verbose:
            tk_train = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Train Epoch {}/{}'.format(epoch + 1, epochs)
            )
        else:
            tk_train = train_loader

        # Make sure the model is set to train mode
        if isinstance(model, AbstractNormalizingFlow):
            model.train(base_mode=train_base)
        else:
            model.train()

        # Training phase
        for inputs, targets in tk_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = nll_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            running_train_loss(loss.item() * train_loader.batch_size)
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_train_hits(hits.item())

        # Initialize the tqdm validation data loader, if verbose is specified
        if verbose:
            tk_val = tqdm(
                val_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
                desc='Validation Epoch {}/{}'.format(epoch + 1, epochs)
            )
        else:
            tk_val = val_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        with torch.no_grad():
            for inputs, targets in tk_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = torch.log_softmax(model(inputs), dim=1)
                loss = nll_loss(outputs, targets)
                running_val_loss(loss.item() * val_loader.batch_size)
                predictions = torch.argmax(outputs, dim=1)
                hits = torch.eq(predictions, targets).sum()
                running_val_hits(hits.item())

        # Get the average train and validation losses and accuracies and print it
        end_time = time.perf_counter()
        train_loss = running_train_loss.average()
        train_accuracy = running_train_hits.average()
        val_loss = running_val_loss.average()
        val_accuracy = running_val_hits.average()
        print('Epoch {}/{} - train_loss: {:.4f}, validation_loss: {:.4f}, '.format(
            epoch + 1, epochs, train_loss, val_loss
        ), end='')
        print('train_acc: {:.1f}%, validation_acc: {:.1f}% [{}s]'.format(
            train_accuracy * 100, val_accuracy * 100, int(end_time - start_time)
        ))

        # Append losses and accuracies to history data
        history['train']['loss'].append(train_loss)
        history['train']['accuracy'].append(train_accuracy)
        history['validation']['loss'].append(val_loss)
        history['validation']['accuracy'].append(val_accuracy)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: {:.4f}'.format(early_stopping.best_loss))
            break

    # Load the best parameters state according to early stopping
    model.load_state_dict(early_stopping.best_state)
    return history


def test_model(
        model,
        data_test,
        setting,
        batch_size=100,
        num_workers=0,
        class_weights=None,
        device=None
):
    """
    Test a Torch model.

    :param model: The model to test.
    :param data_test: The test dataset.
    :param setting: The test setting. It can be either 'generative' or 'discriminative'.
    :param batch_size: The batch size for testing.
    :param num_workers: The number of workers for data loading.
    :param class_weights: The class weights (for im-balanced datasets). Used only if setting='discriminative'.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :return: The mean log-likelihood and two standard deviations if setting='generative'.
             The negative log-likelihood and classification metrics if setting='discriminative'.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: {}'.format(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Move the model to device
    model.to(device)

    # Test the model
    if setting == 'generative':
        return test_generative(model, test_loader, device)
    elif setting == 'discriminative':
        return test_discriminative(model, test_loader, class_weights, device)
    else:
        raise ValueError('Unknown test setting called {}'.format(setting))


def test_generative(model, test_loader, device):
    """
    Test a Torch model in generative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :return: The mean log-likelihood and two standard deviations.
    """
    # Make sure the model is set to evaluation mode
    model.eval()

    test_ll = np.array([])
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            ll = model(inputs).cpu().numpy().flatten()
            test_ll = np.hstack((test_ll, ll))
    mu_ll = np.mean(test_ll)
    sigma_ll = 2.0 * np.std(test_ll) / np.sqrt(len(test_ll))
    return mu_ll, sigma_ll


def test_discriminative(model, test_loader, class_weights, device):
    """
    Test a Torch model in discriminative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param class_weights: The class weights (for im-balanced datasets).
    :param device: The device used for testing.
    :return: The negative log-likelihood and classification report dictionary.
    """
    # Make sure the model is set to evaluation mode
    model.eval()

    # Instantiate the loss
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weight = None
    nll_loss = torch.nn.NLLLoss(weight=weight)

    y_true = []
    y_pred = []
    running_loss = RunningAverageMetric(test_loader.batch_size)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            loss = nll_loss(outputs, targets)
            running_loss(loss.item() * test_loader.batch_size)
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
    return running_loss.average(), metrics.classification_report(y_true, y_pred, output_dict=True)
