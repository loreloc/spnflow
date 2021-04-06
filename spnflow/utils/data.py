import numpy as np
import scipy.stats as stats


class DataFlatten:
    """Data flatten transform for probabilistic learning purposes."""
    def __init__(self):
        """
        Build the data transform.
        """
        self.shape = None

    def fit(self, data):
        """
        Fit the data transform with some data.

        :param data: The data for fitting.
        """
        self.shape = data.shape[1:]

    def forward(self, data):
        """
        Apply the data transform to some data.

        :param data: The data to transform.
        :return: The transformed data.
        """
        return np.reshape(data, [len(data), -1])

    def backward(self, data):
        """
        Apply the backward data transform to some data

        :param data: The data to transform.
        :return: The transformed data.
        """
        return np.reshape(data, [len(data), *self.shape])


class DataStandardizer:
    """Data standardizer for probabilistic learning purposes."""
    def __init__(self, sample_wise=True, flatten=False, epsilon=1e-7, dtype=np.float32):
        """
        Build the data transform.

        :param sample_wise: Sample wise standardization.
        :param flatten: Whether to flatten the data.
        :param epsilon: Epsilon factor for standardization.
        :param dtype: The type for type conversion.
        """
        self.sample_wise = sample_wise
        self.flatten = flatten
        self.epsilon = epsilon
        self.dtype = dtype
        self.mu = None
        self.sigma = None
        self.shape = None

    def fit(self, data):
        """
        Fit the data transform with some data.

        :param data: The data for fitting.
        """
        if self.sample_wise:
            self.mu = np.mean(data, axis=0)
            self.sigma = np.std(data, axis=0)
        else:
            self.mu = np.mean(data)
            self.sigma = np.std(data)
        self.shape = data.shape[1:]

    def forward(self, data):
        """
        Apply the data transform to some data.

        :param data: The data to transform.
        :return: The transformed data.
        """
        data = (data - self.mu) / (self.sigma + self.epsilon)
        if self.flatten:
            data = data.reshape([len(data), -1])
        return data.astype(self.dtype)

    def backward(self, data):
        """
        Apply the backward data transform to some data

        :param data: The data to transform.
        :return: The transformed data.
        """
        if self.flatten:
            data = data.reshape([len(data), *self.shape])
        data = (self.sigma + self.epsilon) * data + self.mu
        return data


def ohe_data(data, domain):
    """
    One-Hot-Encoding function.

    :param data: The 1D data to encode.
    :param domain: The domain to use.
    :return: The One Hot encoded data.
    """
    n_samples = len(data)
    ohe = np.zeros((n_samples, len(domain)))
    ohe[np.equal.outer(data, domain)] = 1
    return ohe


def ecdf_data(data):
    """
    Empirical Cumulative Distribution Function (ECDF).

    :param data: The data.
    :return: The result of the ECDF on data.
    """
    return stats.rankdata(data, method='max') / len(data)


def compute_mean_quantiles(data, n_quantiles):
    """
    Compute the mean quantiles of a dataset (Poon-Domingos).

    :param data: The data.
    :param n_quantiles: The number of quantiles.
    :return: The mean quantiles.
    """
    # Split the dataset in quantiles regions
    data = np.sort(data, axis=0)
    values_per_quantile = np.array_split(data, n_quantiles, axis=0)

    # Compute the mean quantiles
    mean_per_quantiles = [np.mean(x, axis=0) for x in values_per_quantile]
    return np.stack(mean_per_quantiles, axis=0)
