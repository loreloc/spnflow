import scipy.stats as stats
from spnflow.structure.node import Node


class Leaf(Node):
    def __init__(self, scope):
        super().__init__([], [scope] if type(scope) == int else scope)

    def likelihood(self, x):
        pass

    def log_likelihood(self, x):
        pass

    def fit(self, data):
        pass

    def mode(self):
        pass

    def sample(self, size=1):
        pass

    @staticmethod
    def params_count():
        pass


class Bernoulli(Leaf):
    def __init__(self, scope, p=0.5):
        super().__init__(scope)
        self.p = p

    def likelihood(self, x):
        return stats.bernoulli.pmf(x, self.p)

    def log_likelihood(self, x):
        return stats.bernoulli.logpmf(x, self.p)

    def fit(self, data):
        self.p = data.sum().item() / len(data)

    def mode(self):
        return 0 if self.p < 0.5 else 1

    def sample(self, size=1):
        return stats.bernoulli.rvs(self.p, size=size)

    @staticmethod
    def params_count():
        return 1


class Uniform(Leaf):
    def __init__(self, scope, start=0.0, width=1.0):
        super().__init__(scope)
        self.start = start
        self.width = width

    def likelihood(self, x):
        return stats.uniform.pdf(x, self.start, self.width)

    def log_likelihood(self, x):
        return stats.uniform.logpdf(x, self.start, self.width)

    def fit(self, data):
        self.start, self.width = stats.uniform.fit(data)

    def mode(self):
        return self.start

    def sample(self, size=1):
        return stats.uniform.rvs(self.start, self.width, size=size)

    @staticmethod
    def params_count():
        return 2


class Gaussian(Leaf):
    def __init__(self, scope, mean=0.0, stdev=1.0):
        super().__init__(scope)
        self.mean = mean
        self.stdev = stdev

    def likelihood(self, x):
        return stats.norm.pdf(x, self.mean, self.stdev)

    def log_likelihood(self, x):
        return stats.norm.logpdf(x, self.mean, self.stdev)

    def fit(self, data):
        self.mean, self.stdev = stats.norm.fit(data)

    def mode(self):
        return self.mean

    def sample(self, size=1):
        return stats.norm.rvs(self.mean, self.stdev, size=size)

    @staticmethod
    def params_count():
        return 2
