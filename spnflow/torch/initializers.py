import math
import torch


@torch.no_grad()
def quantiles_initializer(tensor, dataset):
    """
    Initialize the tensor using quantiles given a dataset.
    Each parameter of the i-th batch is initialized with the i-th quantile of the data.

    :param tensor: The tensor to initialize.
    :param dataset: The dataset used to compute the quantiles.
    """
    # Load the data
    n_samples = len(dataset)
    sample = dataset[0]
    if isinstance(sample, tuple):
        data = torch.zeros(n_samples, *sample[0].size())
        for i in range(n_samples):
            data[i], _ = dataset[i]
    else:
        data = torch.zeros(n_samples, *sample.size())
        for i in range(n_samples):
            data[i] = dataset[i]

    # Sort the data
    sorted_data, indices = torch.sort(data, dim=0)

    # Initialize the quantiles indices
    n_quantiles = tensor.size(0)
    section_quantiles = [math.floor(n_samples / n_quantiles)] * n_quantiles
    section_quantiles[-1] += n_samples % n_quantiles

    # Compute the mean quantiles
    values_per_quantile = torch.split(sorted_data, section_quantiles, dim=0)
    values_per_quantile = [torch.mean(v, dim=0) for v in values_per_quantile]
    mean_quantiles = torch.stack(values_per_quantile, dim=0)
    tensor.copy_(mean_quantiles)
