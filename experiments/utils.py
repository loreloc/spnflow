import numpy as np
import matplotlib.pyplot as plt

from spnflow.torch.utils import torch_train, torch_test


def collect_results_generative(model, data_train, data_valid, data_test, compute_bpp=False, **kwargs):
    # Train the model
    torch_train(model, data_train, data_valid, setting='generative', **kwargs)

    # Test the model
    (mu_ll, sigma_ll) = torch_test(model, data_test, setting='generative')

    # Compute the bits per pixel, if specified
    if compute_bpp:
        dims = np.prod(data_train.shape[1:])
        bpp = np.log2(256) - (mu_ll / (dims * np.log(2)))
        return mu_ll, sigma_ll, bpp
    else:
        return mu_ll, sigma_ll, None


def collect_results_discriminative(model, data_train, data_valid, data_test, **kwargs):
    # Train the model
    torch_train(model, data_train, data_valid, setting='discriminative', **kwargs)

    # Test the model
    (nll, accuracy) = torch_test(model, data_test, setting='discriminative')
    return nll, accuracy


def collect_samples(model, n_samples):
    samples = model.sample(n_samples).cpu().numpy()
    return samples


def save_grid_images(images, filepath):
    n_rows, n_cols, width, height = images.shape
    canvas = np.zeros([n_rows * width, n_cols * height], dtype=np.uint8)
    for i in range(n_rows):
        for j in range(n_cols):
            px, py = i * width, j * height
            qx, qy = px + width, py + height
            canvas[px:qx, py:qy] = images[i, j]
    plt.imsave(filepath, canvas, cmap='gray')
