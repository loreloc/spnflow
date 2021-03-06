import numpy as np
import scipy.stats as stats

from spnflow.structure.leaf import LeafType


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


class DataDequantizer:
    """Data dequantizer for probabilistic learning purposes."""
    def __init__(self, normalize=True, flatten=False, dtype=np.float32):
        """
        Build the data transform.

        :param normalize: Whether to normalize the data from [0, 255] to [0, 1].
        :param flatten: Whether to flatten the data.
        :param epsilon: Epsilon factor for standardization.
        :param dtype: The type for type conversion.
        """
        self.normalize = normalize
        self.flatten = flatten
        self.dtype = dtype
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
        data = data + np.random.rand(*data.shape)
        if self.normalize:
            data = data / 256.0
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
        if self.normalize:
            data = np.clip(data, 0.0, 1.0) * 255.0
        else:
            data = np.clip(data, 0.0, 255.0)
        return data


class DataTransorms:
    """Data transformer consisting on a chain of transforms."""
    def __init__(self, *transforms):
        assert len(transforms) > 0
        self.transforms = transforms

    def fit(self, data):
        """
        Fit the data transform with some data.

        :param data: The data for fitting.
        """
        for transform in self.transforms:
            transform.fit(data)
            data = transform.forward(data)

    def forward(self, data):
        """
        Apply the data transform to some data.

        :param data: The data to transform.
        :return: The transformed data.
        """
        for transform in self.transforms:
            data = transform.forward(data)
        return data

    def backward(self, data):
        """
        Apply the backward data transform to some data

        :param data: The data to transform.
        :return: The transformed data.
        """
        for transform in reversed(self.transforms):
            data = transform.backward(data)
        return data


def get_data_domains(data, distributions):
    """
    Compute the domains based on the training data and the features distributions.

    :param data: The training data.
    :param distributions: A list of distribution classes.
    :return: A list of domains.
    """
    assert data is not None
    assert distributions is not None

    domains = []
    for i, d in enumerate(distributions):
        col = data[:, i]
        min = np.min(col)
        max = np.max(col)
        if d.LEAF_TYPE == LeafType.DISCRETE:
            domains.append(list(range(max.astype(int) + 1)))
        elif d.LEAF_TYPE == LeafType.CONTINUOUS:
            domains.append([min, max])
        else:
            raise NotImplementedError("Domain for leaf type " + d.LEAF_TYPE.__name__ + " not implemented")
    return domains


def mixed_ohe_data(data, distributions, domains):
    """
    One-Hot-Encoding function, applied on mixed data (both continuous and discrete).

    :param data: The 2D data to encode.
    :param distributions: The given distributions.
    :param domains: The domains to use.
    :return: The One Hot encoded data.
    """
    n_samples, n_features = data.shape
    ohe = []
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:
            ohe.append(ohe_data(data[:, i], domains[i]))
        else:
            ohe.append(data[:, i])
    return np.column_stack(ohe)


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
