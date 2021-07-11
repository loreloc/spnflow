import abc
import numpy as np
import scipy.stats as ss

from enum import Enum
from deeprob.spn.structure.node import Node
from deeprob.spn.utils.data import ohe_data


class LeafType(Enum):
    """
    The type of the distribution leaf. It can be discrete or continuous.
    """
    DISCRETE = 1
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
        super(Leaf, self).__init__([], [scope] if type(scope) == int else scope)

    def em_init(self, random_state):
        """
        Random initialize the leaf's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """
        pass

    def em_step(self, stats, data):
        """
        Compute the parameters after an EM step.

        :param stats: The sufficient statistics of each sample.
        :param data: The data regarding random variables of the leaf.
        :return: A dictionary of new parameters.
        """
        return dict()

    @abc.abstractmethod
    def fit(self, data, domain, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param kwargs: Optional parameters.
        """
        pass

    @abc.abstractmethod
    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        pass

    @abc.abstractmethod
    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        pass

    @abc.abstractmethod
    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        pass

    @abc.abstractmethod
    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        pass

    @abc.abstractmethod
    def params_count(self):
        """
        Get the number of parameters of the distribution leaf.

        :return: The number of parameters.
        """
        pass

    @abc.abstractmethod
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
        super(Bernoulli, self).__init__(scope)
        self.p = p

    def fit(self, data, domain, alpha=0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: Laplace smoothing factor.
        :param kwargs: Optional parameters.
        """
        self.p = (data.sum().item() + alpha) / (len(data) + 2 * alpha)

    def em_init(self, random_state):
        """
        Random initialize the leaf's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """
        self.p = random_state.rand()

    def em_step(self, stats, data):
        """
        Compute the parameters after an EM step.

        :param stats: The sufficient statistics of each sample.
        :param data: The data regarding random variables of the leaf.
        :return: A dictionary of new parameters.
        """
        total_stats = np.sum(stats) + np.finfo(np.float32).eps
        p = np.sum(stats[data == 1]) / total_stats
        return {'p': p}

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return ss.bernoulli.pmf(x, self.p)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        y = ss.bernoulli.logpmf(x, self.p)
        return y

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        z[np.isnan(x)] = 0 if self.p < 0.5 else 1
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        mask = np.isnan(x)
        n_nans = np.count_nonzero(mask)
        z[mask] = ss.bernoulli.rvs(self.p, size=n_nans)
        return z

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


class Categorical(Leaf):
    """
    The Categorical distribution leaf.
    """
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, p=None):
        """
        Initialize a Categorical leaf node given its scope.

        :param scope: The scope of the leaf.
        :param p: The probability of each category.
        """
        super(Categorical, self).__init__(scope)
        self.p = p

    def fit(self, data, domain, alpha=0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: Laplace smoothing factor.
        :param kwargs: Optional parameters.
        """
        self.p = []
        len_data = len(data)
        len_domain = len(domain)
        for c in range(len_domain):
            prob = (len(data[data == c]) + alpha) / (len_data + len_domain * alpha)
            self.p.append(prob)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        mask = np.isnan(x)
        ll = np.ones(shape=(len(x), 1))
        z = x[~mask].astype(np.int64)
        ll[~mask] = ss.multinomial.pmf(ohe_data(z, range(len(self.p))), 1, self.p)
        return ll

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        mask = np.isnan(x)
        ll = np.zeros(shape=(len(x), 1))
        z = x[~mask].astype(np.int64)
        ll[~mask] = ss.multinomial.logpmf(ohe_data(z, range(len(self.p))), 1, self.p)
        return ll

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        z[np.isnan(x)] = np.argmax(self.p)
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        mask = np.isnan(x)
        n_nans = np.count_nonzero(mask)
        z[mask] = np.argmax(ss.multinomial.rvs(1, self.p, size=n_nans), axis=1)
        return z

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


class Isotonic(Leaf):
    """
    The Isotonic distribution leaf.
    """
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, continuous=False, densities=None, breaks=None, mids=None):
        """
        Initialize an Isotonic leaf node given its scope.

        :param scope: The scope of the leaf.
        :param continuous: Flag checking a continuous domain.
        :param densities: The densities.
        :param breaks: The breaks values.
        :param mids: The mids values.
        """
        super(Isotonic, self).__init__(scope)
        self.continuous = continuous
        self.densities = np.array(densities) if densities is not None else None
        self.breaks = np.array(breaks) if breaks is not None else None
        self.mids = np.array(mids) if mids is not None else None

    def fit(self, data, domain, alpha=0.1, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param alpha: Laplace smoothing factor.
        :param kwargs: Optional parameters.
        """
        n_samples, n_features = data.shape
        if self.continuous:
            histogram, self.breaks = np.histogram(data, bins='auto')
            self.mids = ((self.breaks + np.roll(self.breaks, -1)) / 2.0)[:-1]
        else:
            bins = np.array(domain + [domain[-1] + 1])
            histogram, self.breaks = np.histogram(data, bins=bins)
            self.mids = np.array(domain)

        # Apply Laplace smoothing and obtain the densities
        n_bins = len(self.breaks) - 1
        self.densities = (histogram + alpha) / (n_samples + n_bins * alpha)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return np.exp(self.log_likelihood(x))

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        oob = (x < self.breaks[0]) | (x >= self.breaks[-1])
        mask = np.isnan(x) | oob
        ll = np.zeros(shape=(len(x), 1))
        z = np.expand_dims(x[~mask], axis=1)
        j = np.argmax(z < self.breaks, axis=1)
        ll[~mask] = np.log(self.densities[j - 1])
        ll[oob] = -np.inf
        return ll

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        z[np.isnan(x)] = self.mids[np.argmax(self.densities)]
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        mask = np.isnan(x)
        n_nans = np.count_nonzero(mask)
        if self.continuous == LeafType.DISCRETE:
            z[mask] = np.random.choice(self.mids, p=self.densities, size=n_nans)
        else:
            z[mask] = ss.rv_histogram((self.densities, self.breaks)).ppf(
                np.random.rand(n_nans)
            )
        return z

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
        return {
            'continuous': self.continuous,
            'densities': self.densities.tolist(),
            'breaks': self.breaks.tolist(),
            'mids': self.mids.tolist()
        }


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
        super(Uniform, self).__init__(scope)
        self.start = start
        self.width = width

    def fit(self, data, domain, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param kwargs: Optional parameters.
        """
        self.start, self.width = ss.uniform.fit(data)

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return ss.uniform.pdf(x, self.start, self.width)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return ss.uniform.logpdf(x, self.start, self.width)

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        z[np.isnan(x)] = self.start
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        mask = np.isnan(x)
        n_nans = np.count_nonzero(mask)
        z[mask] = ss.uniform.rvs(self.start, self.width, size=n_nans)
        return z

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
        return {
            'start': self.start,
            'width': self.width
        }


class Gaussian(Leaf):
    """
    The Gaussian distribution leaf.
    """
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, mean=0.0, stddev=1.0):
        """
        Initialize a Gaussian leaf node given its scope.

        :param scope: The scope of the leaf.
        :param mean: The mean parameter.
        :param stddev: The standard deviation parameter.
        """
        super(Gaussian, self).__init__(scope)
        self.mean = mean
        self.stddev = stddev

    def fit(self, data, domain, **kwargs):
        """
        Fit the distribution parameters given the domain and some training data.

        :param data: The training data.
        :param domain: The domain of the distribution leaf.
        :param kwargs: Optional parameters.
        """
        self.mean, self.stddev = ss.norm.fit(data)
        self.stddev = max(self.stddev, 1e-5)

    def em_init(self, random_state):
        """
        Random initialize the leaf's parameters for Expectation-Maximization (EM).

        :param random_state: The random state.
        """
        self.mean = 1e-1 * random_state.randn()
        self.stddev = 0.5 + 1e-1 * np.tanh(random_state.randn())

    def em_step(self, stats, data):
        """
        Compute the parameters after an EM step.

        :param stats: The sufficient statistics of each sample.
        :param data: The data regarding random variables of the leaf.
        :return: A dictionary of new parameters.
        """
        total_stats = np.sum(stats) + np.finfo(np.float32).eps
        mean = np.sum(stats * data) / total_stats
        stddev = np.sqrt(np.sum(stats * (data - mean) ** 2.0) / total_stats)
        stddev = max(stddev, 1e-5)
        return {'mean': mean, 'stddev': stddev}

    def likelihood(self, x):
        """
        Compute the likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting likelihood.
        """
        return ss.norm.pdf(x, self.mean, self.stddev)

    def log_likelihood(self, x):
        """
        Compute the logarithmic likelihood of the distribution leaf given some input.

        :param x: The inputs.
        :return: The resulting log likelihood.
        """
        return ss.norm.logpdf(x, self.mean, self.stddev)

    def mpe(self, x):
        """
        Compute the maximum at posteriori values.

        :return: The distribution's maximum at posteriori values.
        """
        z = np.copy(x)
        z[np.isnan(x)] = self.mean
        return z

    def sample(self, x):
        """
        Sample from the leaf distribution.

        :param x: The samples with possible NaN values.
        :return: The completed samples.
        """
        z = np.copy(x)
        mask = np.isnan(x)
        n_nans = np.count_nonzero(mask)
        z[mask] = ss.norm.rvs(self.mean, self.stddev, size=n_nans)
        return z

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
        return {
            'mean': self.mean,
            'stddev': self.stddev
        }
