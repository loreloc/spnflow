import time
import torch
import numpy as np
from spnflow.torch.callbacks import EarlyStopping


def torch_train_generative(
        model,
        data_train,
        data_val,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optim=torch.optim.Adam,
        init_args={},
        device=None,
):
    """
    Train a Torch model by maximizing the log-likelihood.

    :param model: The model to train.
    :param data_train: The train data.
    :param data_val: The validation data.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param patience: The number of consecutive epochs to wait until no improvements of the validation loss occurs.
    :param optim: The optimizer to use.
    :param init_args: The parameters to pass to the initializers of the model.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
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

    # Call the model initializer
    model.apply_initializers(**init_args)

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

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
            loss = -torch.mean(log_likelihoods)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.apply_constraints()
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

    return history


def torch_test_generative(
        model,
        data_test,
        batch_size=100,
        device=None
    ):
    """
    Test a Torch model by its log-likelihood.

    :param model: The model to test.
    :param data_test: The test data.
    :param batch_size: The batch size for testing.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

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
