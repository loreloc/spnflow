import torch
import numpy as np


@torch.no_grad()
def quantiles_initializer(tensor, data):
    """
    Initialize the tensor using quantiles given some data.
    Each parameter of the i-th batch is initialized with the i-th quantile of the data.

    :param tensor: The tensor to initialize.
    :param data: The data used to compute the quantiles.
    """
    # Sort the data
    n_samples = len(data)
    sorted_data = np.sort(data, axis=0)

    # Initialize the quantiles indices
    n_quantiles = tensor.size(0)
    idx_quantiles = np.ceil(n_samples * np.arange(1, n_quantiles) / n_quantiles).astype(np.int)

    # Compute the mean quantiles
    values_per_quantile = np.split(sorted_data, idx_quantiles, axis=0)
    values_per_quantile = [np.mean(v, axis=0) for v in values_per_quantile]
    mean_quantiles = np.stack(values_per_quantile, axis=0)
    tensor.copy_(torch.tensor(mean_quantiles))
