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

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
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

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'p': self.p}


class Multinomial(Leaf):
    """
    The Multinomial (or Categorical) distribution leaf.
    """

    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, p=[0.5, 0.5]):
        """
        Initialize a Multinomial leaf node given its scope.

        :param scope: The scope of the leaf.
        :param k: The number of classes.
        """
        super().__init__(scope)
        self.p = p

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.p = []
        len_data = len(data)
        for c in range(len(domain)):
            self.p.append(len(data[data == c]) / len_data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        x_len = len(x)
        k = len(self.p)
        z = np.zeros((x_len, k))
        z[np.arange(x_len), x.astype(int)] = 1
        return stats.multinomial.pmf(z, 1, self.p)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        x_len = len(x)
        k = len(self.p)
        z = np.zeros((x_len, k))
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
        return len(self.p)

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'p': self.p}


class Poisson(Leaf):
    """
    The Poisson distribution leaf.
    """

    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, mu=1.0):
        """
        Initialize a Poisson leaf node given its scope.

        :param scope: The scope of the leaf.
        :param mu: The mu parameter.
        """
        super().__init__(scope)
        self.mu = mu

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        self.mu = np.mean().item()

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return stats.poisson.pmf(x, self.mu)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return stats.poisson.logpmf(x, self.mu)

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return np.floor(self.mu)

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        return stats.poisson.rvs(self.mu)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 1

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'mu': self.mu}


class Isotonic(Leaf):
    """
    The Isotonic distribution leaf.
    """

    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, meta, densities=[], breaks=[], mids=[]):
        """
        Initialize an Isotonic leaf node given its scope.

        :param scope: The scope of the leaf.
        :param meta: The meta type.
        :param delta: The smoothing factor.
        """
        super().__init__(scope)
        self.meta = meta
        self.densities = densities
        self.breaks = breaks
        self.mids = mids

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        n_samples, _ = data.shape
        if self.meta == LeafType.DISCRETE:
            bins = np.array([d for d in domain] + [domain[-1] + 1])
            self.densities, self.breaks = np.histogram(data, bins=bins, density=True)
            self.mids = np.array(domain)
        else:
            self.densities, self.breaks = np.histogram(data, bins='auto', density=True)
            self.mids = ((self.breaks + np.roll(self.breaks, -1)) / 2.0)[:-1]

        # Apply Laplace smoothing
        n_bins = len(self.breaks) - 1
        self.densities = (self.densities * n_samples + 1.0) / (n_samples + n_bins)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        n_samples = len(x)
        l = np.full(n_samples, np.finfo(float).eps)
        for i in range(n_samples):
            j = np.searchsorted(self.breaks, x[i], side='right')
            if j == 0 or j == len(self.breaks):
                continue
            l[i] = self.densities[j - 1]
        return l

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return np.log(self.likelihood(x))

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return self.mids[np.argmax(self.densities)]

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        if self.meta == LeafType.DISCRETE:
            return np.random.choice(self.mids, p=self.densities, size=size)
        else:
            q = stats.uniform.rvs(size=size)
            return stats.rv_histogram((self.densities, self.breaks)).ppf(q)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 1 + len(self.densities) + len(self.breaks) + len(self.mids)

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'meta': self.meta, 'densities': self.densities, 'breaks': self.breaks, 'mids': self.mids}


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

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'start': self.start, 'width': self.width}


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

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'mean': self.mean, 'stdev': self.stdev}


class Gamma(Leaf):
    """
    The Gamma distribution leaf.
    """

    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, alpha=1.0, loc=0.0, beta=2.0):
        """
        Initialize a Gamma leaf node given its scope.

        :param scope: The scope of the leaf.
        :param alpha: The alpha parameter.
        :param beta: The beta parameter.
        """
        super().__init__(scope)
        self.alpha = alpha
        self.loc = loc
        self.beta = beta

    def fit(self, data, domain):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        """
        assert np.any(data < 0.0), "Cannot fit Gamma distribution leaf with negative data"
        self.alpha, self.loc, self.beta = stats.gamma.fit(data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return stats.gamma.pdf(x, self.alpha, self.loc, self.beta)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return stats.gamma.logpdf(x, self.alpha, self.loc, self.beta)

    def mode(self):
        """
        Compute the mode of the distribution.

        :return: The distribution's mode.
        """
        return (self.alpha - 1.0) * self.beta

    def sample(self, size=1):
        """
        Sample from the leaf distribution.

        :param size: The number of samples.
        :return: Some samples.
        """
        return stats.gamma.rvs(self.alpha, self.loc, self.beta, size=size)

    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        return 3

    def params_dict(self):
        """
        Get a dictionary representation of the distribution parameters.

        :return: A dictionary containing the distribution parameters.
        """
        return {'loc': self.loc, 'alpha': self.alpha, 'beta': self.beta}
