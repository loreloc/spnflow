import scipy.stats as stats


def random_rows(data, a=2.0, b=2.0):
    """
    Choose a binary partition horizontally randomly.
    The proportion of the split is sampled from a beta distribution.

    :param data: The data.
    :param a: The alpha parameter of the beta distribution.
    :param b: The beta parameter of the beta distribution.
    :return: A binary partition.
    """
    n_samples, _ = data.shape
    p = stats.beta.rvs(a, b)
    return stats.bernoulli.rvs(p, size=n_samples)


def random_cols(data, a=2.0, b=2.0):
    """
    Choose a binary partition horizontally randomly.
    The proportion of the split is sampled from a beta distribution.

    :param data: The data.
    :param a: The alpha parameter of the beta distribution.
    :param b: The beta parameter of the beta distribution.
    :return: A binary partition.
    """
    _, n_features = data.shape
    p = stats.beta.rvs(a, b)
    return stats.bernoulli.rvs(p, size=n_features)
