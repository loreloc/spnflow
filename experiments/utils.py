import numpy as np
import matplotlib.pyplot as plt

from spnflow.algorithms.inference import log_likelihood
from spnflow.torch.routines import train, torch_test


def evaluate_log_likelihoods(spn, data, batch_size=2048):
    n_samples = len(data)
    ll = np.zeros(n_samples)
    for i in range(0, n_samples - batch_size, batch_size):
        ll[i:i + batch_size] = log_likelihood(spn, data[i:i + batch_size])
    n_remaining_samples = n_samples % batch_size
    if n_remaining_samples > 0:
        ll[-n_remaining_samples:] = log_likelihood(spn, data[-n_remaining_samples:])
    mean_ll = np.mean(ll)
    stddev_ll = 2.0 * np.std(ll) / np.sqrt(n_samples)
    return mean_ll, stddev_ll


def collect_results_generative(model, data_train, data_valid, data_test, compute_bpp=False, **kwargs):
    # Train the model
    train(model, data_train, data_valid, setting='generative', **kwargs)

    # Test the model
    (mu_ll, sigma_ll) = torch_test(model, data_test, setting='generative')

    # Compute the bits per pixel, if specified
    if compute_bpp:
        dims = np.prod(data_train.shape[1:])
        bpp = -(mu_ll / np.log(2)) / dims
        return mu_ll, sigma_ll, bpp
    else:
        return mu_ll, sigma_ll, None


def collect_results_discriminative(model, data_train, data_valid, data_test, **kwargs):
    # Train the model
    train(model, data_train, data_valid, setting='discriminative', **kwargs)

    # Test the model
    nll, metrics = torch_test(model, data_test, setting='discriminative')
    return nll, metrics


def collect_samples(model, n_samples):
    # Make sure to switch to evaluation mode
    model.eval()
    samples = model.sample(n_samples).cpu().numpy()
    return samples


def save_grid_images(images, filepath):
    images = images.copy()
    images[images < 0] = 0
    images[images > 255] = 255
    images = images.astype(np.uint8)
    n_rows, n_cols, channels, width, height = images.shape
    canvas = np.zeros([n_rows * width, n_cols * height, channels], dtype=np.uint8)
    for i in range(n_rows):
        for j in range(n_cols):
            px, py = i * width, j * height
            qx, qy = px + width, py + height
            canvas[px:qx, py:qy] = np.transpose(images[i, j], axes=[1, 2, 0])
    if channels == 1:
        canvas = np.squeeze(canvas, axis=-1)
    plt.imsave(filepath, canvas, cmap='gray')
