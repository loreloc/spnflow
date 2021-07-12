def random_rows(data, distributions, domains, random_state, a=2.0, b=2.0):
    """
    Choose a binary partition horizontally randomly.
    The proportion of the split is sampled from a beta distribution.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param random_state: The random state.
    :param a: The alpha parameter of the beta distribution.
    :param b: The beta parameter of the beta distribution.
    :return: A binary partition.
    """
    n_samples, _ = data.shape
    p = random_state.beta(a, b)
    return random_state.binomial(1, p, size=n_samples)


def random_cols(data, distributions, domains, random_state, a=2.0, b=2.0):
    """
    Choose a binary partition vertically randomly.
    The proportion of the split is sampled from a beta distribution.

    :param data: The data.
    :param distributions: The data distributions (not used).
    :param domains: The data domains (not used).
    :param random_state: The random state.
    :param a: The alpha parameter of the beta distribution.
    :param b: The beta parameter of the beta distribution.
    :return: A binary partition.
    """
    _, n_features = data.shape
    p = random_state.beta(a, b)
    return random_state.binomial(1, p, size=n_features)