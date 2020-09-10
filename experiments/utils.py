import os
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spnflow.torch.utils import torch_train_generative, torch_test_generative


def logit(data, alpha=0.05):
    data = alpha + (1.0 - 2.0 * alpha) * data
    return np.log(data / (1.0 - data))


def delogit(data, alpha=0.05):
    x = 1.0 / (1.0 + np.exp(-data))
    return (x - alpha) / (1.0 - 2.0 * alpha)


def dequantize(data, rand_state):
    return data + rand_state.rand(*data.shape)


def collect_results(
        dataset,
        info,
        model,
        data_train,
        data_val,
        data_test,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optim=torch.optim.Adam,
        init_args={},
    ):
    """
    Train and test the model on given data.

    :param dataset: The dataset's name.
    :param info: Information string about the experiment.
    :param model: The model to train and test.
    :param data_train: The train data.
    :param data_val: The validation data.
    :param data_test: The test data.
    :param lr: The learning rate.
    :param batch_size: The batch size.
    :param epochs: The number of epochs.
    :param patience: The early stopping patience.
    :param optim: The optimizer class to use.
    :param init_args: The arguments to pass to the model's parameter initializer.
    """
    # Train the model
    history = torch_train_generative(model, data_train, data_val, lr, batch_size, epochs, patience, optim, init_args)

    # Test the model
    (mu_ll, sigma_ll) = torch_test_generative(model, data_test, batch_size)

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Mean Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two StdDev. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')

    # Plot the training history
    filepath = os.path.join('histories', dataset + '_' + info + '.png')
    plt.xlim(0, epochs)
    plt.plot(history['train'])
    plt.plot(history['validation'])
    plt.title('Log-Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(filepath)
    plt.clf()


def collect_samples(
        dataset,
        info,
        model,
        image_func=None,
        n_samples=100,
        n_grid=(8, 8),
    ):
    """
    Sample some samples from the model.

    :param dataset: The dataset's name.
    :param info: The information string about the experiment.
    :param model: The model which to sample from.
    :param image_func: The function that convert the samples to images.
    :param n_samples: The number of samples. This is ignored if image_func is not None.
    :param n_grid: The grid of samples shape. This is ignore if image_func is None.
    """
    # Initialize the results filepath
    filepath = os.path.join('samples', dataset + '_' + info)

    # If image_func is specified, save the samples as images, otherwise save them in CSV format
    if image_func:
        rows, cols = n_grid
        n_samples = rows * cols
        samples = model.sample(n_samples).cpu().numpy()
        images = image_func(samples)
        torchvision.utils.save_image(torch.tensor(images), filepath + '.png', nrow=cols, padding=0, pad_value=255)
    else:
        samples = model.sample(n_samples).cpu().numpy()
        pd.DataFrame(samples).to_csv(filepath + '.csv')