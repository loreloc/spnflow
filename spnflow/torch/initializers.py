import torch
import numpy as np


def quantiles_initializer(tensor, data):
    """
    Initialize the tensor using quantiles given some data.
    Each parameter of the i-th batch is initialized with the i-th quantile of the data.

    :param tensor: The tensor to initialize.
    :param data: The data used to compute the quantiles.
    """
    # Initialize the tensor using quantiles
    with torch.no_grad():
        n_quantiles = tensor.size(0)
        quantiles = np.quantile(data, np.arange(n_quantiles) / n_quantiles, axis=0)
        tensor.copy_(torch.tensor(quantiles))
