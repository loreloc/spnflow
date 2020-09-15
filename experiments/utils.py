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
    filepath = os.path.join('results', 'generative', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Mean Log-Likelihood: ' + str(mu_ll) + '\n')
        file.write('Two StdDev. Log-Likelihood: ' + str(2.0 * sigma_ll) + '\n')

    # Plot the training history
    filepath = os.path.join('histories', 'generative', dataset + '_' + info + '.png')
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
    filepath = os.path.join('results', 'discriminative', dataset + '_' + info + '.txt')
    with open(filepath, 'w') as file:
        file.write(dataset + ': ' + info + '\n')
        file.write('Negative Log-Likelihood: ' + str(nll) + '\n')
        file.write('Accuracy: ' + str(accuracy) + '\n')

    # Plot the training history (loss and accuracy)
    filepath = os.path.join('histories', 'discriminative', dataset + '_' + info + '.png')
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
    :param transform: The function that converts the tensor samples to images. It can be None.
    """
    # Initialize the results filepath
    filepath = os.path.join('samples', dataset + '_' + info)

    # If transform is specified, save the samples as images, otherwise save them in CSV format
    if transform:
        rows, cols = (1, n_samples) if isinstance(n_samples, int) else n_samples
        n_samples = rows * cols
        samples = model.sample(n_samples).cpu()
        images = torch.stack([transform(x) for x in samples], dim=0)
        torchvision.utils.save_image(images, filepath + '.png', nrow=cols, padding=0)
    else:
        n_samples = np.prod(n_samples)
        samples = model.sample(n_samples).cpu().numpy()
        pd.DataFrame(samples).to_csv(filepath + '.csv')


def collect_completions(dataset, info, model, data_test, n_samples, transform=None, rand_state=None):
    """
    Complete some random images from the test set.

    :param dataset: The dataset's name.
    :param info: The information string about the experiment.
    :param model: The model used for completions.
    :param data_test: The test dataset.
    :param n_samples: The number of samples to complete for each mask kind.
    :param transform: The function that converts the tensor samples to images. It can be None.
    :param rand_state: The random state used to get samples to complete. It can be None.
    """
    # Initialize the results filepath
    filepath = os.path.join('completions', dataset + '_' + info + '.png')

    # Get some random samples from the test set
    if rand_state is None:
        rand_state = np.random.RandomState(42)
    idx_samples = rand_state.choice(np.arange(len(data_test)), n_samples, replace=False)
    samples = torch.stack([data_test[i] for i in idx_samples], dim=0)
    n_samples, n_channels, image_h, image_w = samples.size()

    # Compute the masked samples (left, right, top, bottom patterns)
    idx_lef = torch.tensor([j for j in range(image_w // 2)])
    idx_rig = torch.tensor([j for j in range(image_w // 2, image_w)])
    idx_top = torch.tensor([i for i in range(image_h // 2)])
    idx_bot = torch.tensor([i for i in range(image_h // 2, image_h)])
    samples_lef = torch.index_fill(samples, dim=3, index=idx_lef, value=np.nan)
    samples_rig = torch.index_fill(samples, dim=3, index=idx_rig, value=np.nan)
    samples_top = torch.index_fill(samples, dim=2, index=idx_top, value=np.nan)
    samples_bot = torch.index_fill(samples, dim=2, index=idx_bot, value=np.nan)

    # Compute the maximum at posteriori estimates for image completion
    model = model.cpu()
    samples = torch.cat([samples_lef, samples_rig, samples_top, samples_bot], dim=0)
    samples_full = model.mpe(samples)

    # Apply the transformation, if specified
    if transform:
        samples_full = torch.stack([transform(x) for x in samples_full], dim=0)

    # Save the completed images
    torchvision.utils.save_image(samples_full, filepath, nrow=n_samples, padding=0)
