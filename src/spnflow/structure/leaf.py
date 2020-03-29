from enum import Enum
import numpy as np
import scipy.stats as stats
from spnflow.structure.node import Node


class LeafType(Enum):
    DISCRETE = 1,
    CONTINUOUS = 2


class Leaf(Node):
    def __init__(self, scope):
        super().__init__([], [scope] if type(scope) == int else scope)

    def fit(self, data, domain):
        pass

    def likelihood(self, x):
        pass

    def log_likelihood(self, x):
        pass

    def mode(self):
        pass

    def sample(self, size=1):
        pass

    def params_count(self):
        pass


class Bernoulli(Leaf):
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, p=0.5):
        super().__init__(scope)
        self.p = p

    def fit(self, data, domain):
        self.p = data.sum().item() / len(data)

    def likelihood(self, x):
        return stats.bernoulli.pmf(x, self.p)

    def log_likelihood(self, x):
        y = stats.bernoulli.logpmf(x, self.p)
        return y

    def mode(self):
        return 0 if self.p < 0.5 else 1

    def sample(self, size=1):
        return stats.bernoulli.rvs(self.p, size=size)

    def params_count(self):
        return 1


class Multinomial(Leaf):
    LEAF_TYPE = LeafType.DISCRETE

    def __init__(self, scope, k=2):
        super().__init__(scope)
        self.k = k
        self.p = [1.0 / k for i in range(k)]

    def fit(self, data, domain):
        self.k = len(domain)
        self.p = []
        len_data = len(data)
        for c in range(self.k):
            self.p.append(len(data[data == c]) / len_data)

    def likelihood(self, x):
        x_len = len(x)
        z = np.zeros((x_len, self.k))
        z[np.arange(x_len), x.astype(int)] = 1
        return stats.multinomial.pmf(z, 1, self.p)

    def log_likelihood(self, x):
        x_len = len(x)
        z = np.zeros((x_len, self.k))
        z[np.arange(x_len), x.astype(int)] = 1
        return stats.multinomial.logpmf(z, 1, self.p)

    def mode(self):
        return np.argmax(self.p)

    def sample(self, size=1):
        s = stats.multinomial.rvs(1, self.p, size=size)
        return np.argmax(s, axis=1)

    def params_count(self):
        return 1 + self.k


class Uniform(Leaf):
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, start=0.0, width=1.0):
        super().__init__(scope)
        self.start = start
        self.width = width

    def fit(self, data, domain):
        self.start, self.width = stats.uniform.fit(data)

    def likelihood(self, x):
        return stats.uniform.pdf(x, self.start, self.width)

    def log_likelihood(self, x):
        return stats.uniform.logpdf(x, self.start, self.width)

    def mode(self):
        return self.start

    def sample(self, size=1):
        return stats.uniform.rvs(self.start, self.width, size=size)

    def params_count(self):
        return 2


class Gaussian(Leaf):
    LEAF_TYPE = LeafType.CONTINUOUS

    def __init__(self, scope, mean=0.0, stdev=1.0):
        super().__init__(scope)
        self.mean = mean
        self.stdev = stdev

    def fit(self, data, domain):
        self.mean, self.stdev = stats.norm.fit(data)
        if np.isclose(self.stdev, 0.0):
            self.stdev = 1e-8

    def likelihood(self, x):
        return stats.norm.pdf(x, self.mean, self.stdev)

    def log_likelihood(self, x):
        return stats.norm.logpdf(x, self.mean, self.stdev)

    def mode(self):
        return self.mean

    def sample(self, size=1):
        return stats.norm.rvs(self.mean, self.stdev, size=size)

    def params_count(self):
        return 2
