from spnflow.structure.leaf import Isotonic


def get_learn_leaf_method(learn_leaf):
    """
    Get the learn leaf method.

    :param learn_leaf: The learn leaf method string to use.
    :return: A learn leaf function.
    """
    if learn_leaf == 'mle':
        return learn_mle
    elif learn_leaf == 'isotonic':
        return learn_isotonic
    else:
        raise NotImplementedError("Unknow learn leaf method called " + learn_leaf)


def learn_mle(data, distribution, domain, scope, alpha=0.1):
    """
    Learn a leaf using Maximum Likelihood Estimate (MLE)

    :param data: The data.
    :param distribution: The distribution of the random variable.
    :param domain: The domain of the random variable.
    :param scope: The scope of the leaf.
    :param alpha: Laplace smoothing factor.
    :return: A leaf distribution with parameters learned by MLE.
    """
    leaf = distribution(scope)
    leaf.fit(data, domain, alpha=alpha)
    return leaf


def learn_isotonic(data, distribution, domain, scope):
    """
    Learn a leaf using Isotonic method.

    :param data: The data.
    :param distribution: The distribution of the random variable.
    :param domain: The domain of the random variable.
    :param scope: The scope of the leaf.
    :return:
    """
    leaf = Isotonic(scope, distribution.LEAF_TYPE)
    leaf.fit(data, domain)
    return leaf
