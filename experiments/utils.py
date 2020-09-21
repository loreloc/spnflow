import os
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from spnflow.torch.utils import torch_train, torch_test


def get_activation_class(name):
    if name == 'relu':
        return torch.nn.ReLU
    elif name == 'tanh':
        return torch.nn.Tanh,
    elif name == 'sigmoid':
        return torch.nn.Sigmoid
    else:
        raise ValueError


def get_experiment_filename(name, settings):
    return '%s-%s-%s' % (name, settings['dataset'], datetime.now().strftime('%m%d%H%M'))


def collect_results_generative(name, settings, model, data_train, data_val, data_test, bpp=False, **kwargs):
    # Train the model
    history = torch_train(model, data_train, data_val, setting='generative', **kwargs)

    # Test the model
    (mu_ll, sigma_ll) = torch_test(model, data_test, setting='generative')

    # Compute the filename string
    filename = get_experiment_filename(name, settings)

    # Compute the bits per pixel, if specified
    if bpp:
        dims = np.prod(data_train[0].size())
        bpp = np.log2(256) - (mu_ll / (dims * np.log(2)))
    else:
        bpp = None

    # Save the results to file
    filepath = os.path.join(name, 'results')
    os.makedirs(filepath, exist_ok=True)
    results = {'log_likelihoods': {'mean': mu_ll, 'stddev': sigma_ll}, 'bits_per_pixel': bpp, 'settings': settings}
    with open(os.path.join(filepath, filename + '.json'), 'w') as file:
        json.dump(results, file, indent=4)

    # Plot the training history
    filepath = os.path.join(name, 'histories')
    os.makedirs(filepath, exist_ok=True)
    plt.plot(history['train'])
    plt.plot(history['validation'])
    plt.title('Log-Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, filename + '.png'))
    plt.clf()


def collect_results_discriminative(name, settings, model, data_train, data_val, data_test, **kwargs):
    # Train the model
    history = torch_train(model, data_train, data_val, setting='discriminative', **kwargs)

    # Test the model
    (nll, accuracy) = torch_test(model, data_test, setting='discriminative')

    # Compute the filename string
    filename = get_experiment_filename(name, settings)

    # Save the results to file
    filepath = os.path.join(name, 'results')
    os.makedirs(filepath, exist_ok=True)
    results = {'negative_log_likelihood': nll, 'accuracy:': accuracy, 'settings': settings}
    with open(os.path.join(filepath, filename + '.json'), 'w') as file:
        json.dump(results, file, indent=4)

    # Plot the training history
    filepath = os.path.join(name, 'histories')
    os.makedirs(filepath, exist_ok=True)
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
    plt.savefig(os.path.join(filepath, filename + '.png'))
    plt.clf()


def collect_samples(name, settings, model, n_samples, inv_transform=None):
    filepath = os.path.join(name, 'samples')
    os.makedirs(filepath, exist_ok=True)

    # Compute the filename string
    filename = get_experiment_filename(name, settings)

    # If transform is specified, save the samples as images, otherwise save them in CSV format
    if inv_transform:
        samples = model.sample(n_samples ** 2).cpu()
        images = torch.stack([inv_transform(x) for x in samples], dim=0)
        torchvision.utils.save_image(images, os.path.join(filepath, filename + '.png'), nrow=n_samples, padding=0)
    else:
        n_samples = np.prod(n_samples)
        samples = model.sample(n_samples).cpu().numpy()
        pd.DataFrame(samples).to_csv(os.path.join(filepath, filename + '.csv'))


def collect_completions(name, settings, model, data_test, n_samples, inv_transform=None, rand_state=None):
    # Initialize the results filepath
    filename = get_experiment_filename(name, settings)

    # Get some random samples from the test set
    if rand_state is None:
        rand_state = np.random.RandomState(42)
    idx_samples = rand_state.choice(np.arange(len(data_test)), n_samples, replace=False)
    samples = torch.stack([data_test[i] for i in idx_samples], dim=0)
    sample_size = samples.size()[1:]
    target_size = inv_transform(samples[0]).size()
    samples = torch.reshape(samples, [-1, *target_size])
    _, _, image_h, image_w = samples.size()

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
    samples = torch.reshape(samples, [-1, *sample_size])
    samples_full = model.mpe(samples)

    # Apply the transformation, if specified
    if inv_transform:
        samples_full = torch.stack([inv_transform(x) for x in samples_full], dim=0)

    # Save the completed images
    filepath = os.path.join(name, 'completions')
    os.makedirs(filepath, exist_ok=True)
    torchvision.utils.save_image(
        samples_full, os.path.join(filepath, filename + '.png'), nrow=n_samples, padding=0
    )
