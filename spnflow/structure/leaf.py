import numpy as np
import scipy.stats as stats
from enum import Enum
from spnflow.structure.node import Node


class LeafType(Enum):
    """
    The type of the distribution leaf. It can be discrete or continuous.
    """
    DISCRETE = 1,
    CONTINUOUS = 2


class Leaf(Node):
    """
    Distribution leaf base class.
    """
    def __init__(self, scope):
        """
        Initialize a leaf node given its scope.

        :param scope: The scope of the leaf.
        """
        super().__init__([], [scope] if type(scope) == int else scope)

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        pass

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        pass

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        pass

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        pass

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        pass

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        pass


class Bernoulli(Leaf):
    """
    The Bernoulli distribution leaf.
    """

    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, p=0.5):
        """
        Initialize a Bernoulli leaf node given its scope.

        :param scope: The scope of the leaf.
        :param p: The Bernoulli probability.
        """
        super().__init__(scope)
        self.p = p

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.p = data.sum().item() / len(data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return stats.bernoulli.pmf(x, self.p)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        y = stats.bernoulli.logpmf(x, self.p)
        return y

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return 0 if self.p < 0.5 else 1

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        return stats.bernoulli.rvs(self.p, size=size)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 1


class Multinomial(Leaf):
    """
    The Multinomial (or Categorical) distribution leaf.
    """

    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, k=2):
        """
        Initialize a Multinomial leaf node given its scope.

        :param scope: The scope of the leaf.
        :param k: The number of classes.
        """
        super().__init__(scope)
        self.k = k
        self.p = [1.0 / k for i in range(k)]

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.k = len(domain)
        self.p = []
        len_data = len(data)
        for c in range(self.k):
            self.p.append(len(data[data == c]) / len_data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        x_len = len(x)
        z = np.zeros((x_len, self.k))
        z[np.arange(x_len), x.astype(int)] = 1
        return stats.multinomial.pmf(z, 1, self.p)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        x_len = len(x)
        z = np.zeros((x_len, self.k))
        z[np.arange(x_len), x.astype(int)] = 1
        return stats.multinomial.logpmf(z, 1, self.p)

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return np.argmax(self.p)

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        s = stats.multinomial.rvs(1, self.p, size=size)
        return np.argmax(s, axis=1)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 1 + self.k


class Uniform(Leaf):
    """
    The Uniform distribution leaf.
    """

    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, start=0.0, width=1.0):
        """
        Initialize an Uniform leaf node given its scope.

        :param scope: The scope of the leaf.
        :param start: The start of the uniform distribution.
        :param width: The width of the uniform distribution.
        """
        super().__init__(scope)
        self.start = start
        self.width = width

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.start, self.width = stats.uniform.fit(data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return stats.uniform.pdf(x, self.start, self.width)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return stats.uniform.logpdf(x, self.start, self.width)

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return self.start

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        return stats.uniform.rvs(self.start, self.width, size=size)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 2


class Gaussian(Leaf):
    """
    The Gaussian distribution leaf.
    """

    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, mean=0.0, stdev=1.0):
        """
        Initialize a Gaussian leaf node given its scope.

        :param scope: The scope of the leaf.
        :param mean: The mean parameter.
        :param stdev: The standard deviation parameter.
        """
        super().__init__(scope)
        self.mean = mean
        self.stdev = stdev

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.mean, self.stdev = stats.norm.fit(data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return stats.norm.pdf(x, self.mean, self.stdev)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return stats.norm.logpdf(x, self.mean, self.stdev)

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return self.mean

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        return stats.norm.rvs(self.mean, self.stdev, size=size)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 2
