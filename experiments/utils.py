import os
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spnflow.torch.utils import torch_train_generative, torch_test_generative
from spnflow.torch.utils import torch_train_discriminative, torch_test_discriminative


def collect_results_generative(dataset, info, model, data_train, data_val, data_test, **kwargs):
    """
    Train and test the model on given data (generative setting).

    :param dataset: The dataset's name.
    :param info: Information string about the experiment.
    :param model: The model to train and test.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param data_test: The test dataset.
    :param kwargs: Other arguments to pass to the train routine. See `torch_train_generative` docs.
    """
    # Train the model
    history = torch_train_generative(model, data_train, data_val, **kwargs)

    # Test the model
    (mu_ll, sigma_ll) = torch_test_generative(model, data_test)

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + 'gen' + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Mean Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two StdDev. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')

    # Plot the training history
    filepath = os.path.join('histories', dataset + '_' + 'gen' + '_' + info + '_' + '.png')
    plt.plot(history['train'])
    plt.plot(history['validation'])
    plt.title('Log-Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(filepath)
    plt.clf()


def collect_results_discriminative(dataset, info, model, data_train, data_val, data_test, **kwargs):
    """
    Train and test the model on given data (discriminative setting).

    :param dataset: The dataset's name.
    :param info: Information string about the experiment.
    :param model: The model to train and test.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param data_test: The test dataset.
    :param kwargs: Other arguments to pass to the train routine. See `torch_train_discriminative` docs.
    """
    # Train the model
    history = torch_train_discriminative(model, data_train, data_val, **kwargs)

    # Test the model
    (nll, accuracy) = torch_test_discriminative(model, data_test)

    # Save the results to file
    filepath = os.path.join('results', dataset + '_' + 'dis' + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Negative Log-Likelihood: ' + str(nll) + '\n')
        file.write('Accuracy: ' + str(accuracy) + '\n')

    # Plot the training history (loss and accuracy)
    filepath = os.path.join('histories', dataset + '_' + 'dis' + '_' + info + '.png')
    plt.subplot(211)
    plt.plot(history['train']['loss'])
    plt.plot(history['validation']['loss'])
    plt.title('Negative Log-Likelihood')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.subplot(212)
    plt.plot(history['train']['accuracy'])
    plt.plot(history['validation']['accuracy'])
    plt.title('Accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


def collect_samples(dataset, info, model, n_samples, transform=None):
    """
    Sample some samples from the model.

    :param dataset: The dataset's name.
    :param info: The information string about the experiment.
    :param model: The model which to sample from.
    :param n_samples: The number of samples. It can be an integer or a integer pair.
    :param transform: The function that convert the tensor samples to images.
    """
    # Initialize the results filepath
    filepath = os.path.join('samples', dataset + '_' + info)

    # If transform is specified, save the samples as images, otherwise save them in CSV format
    if transform:
        rows, cols = (1, n_samples) if isinstance(n_samples, int) else n_samples
        n_samples = rows * cols
        samples = model.sample(n_samples).cpu()
        images = torch.stack(list(map(transform, torch.unbind(samples, dim=0))), dim=0)
        torchvision.utils.save_image(images, filepath + '.png', nrow=cols, padding=0, pad_value=255)
    else:
        n_samples = np.prod(n_samples)
        samples = model.sample(n_samples).cpu().numpy()
        pd.DataFrame(samples).to_csv(filepath + '.csv')
