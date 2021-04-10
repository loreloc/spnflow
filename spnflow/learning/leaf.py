import numpy as np

from spnflow.structure.node import Mul
from spnflow.structure.leaf import LeafType, Bernoulli, Isotonic
from spnflow.structure.cltree import BinaryCLTree


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
    elif learn_leaf == 'cltree':
        return learn_cltree
    else:
        raise NotImplementedError("Unknown learn leaf method called " + learn_leaf)


def learn_mle(data, distributions, domains, scope, alpha=0.1):
    """
    Learn a leaf using Maximum Likelihood Estimate (MLE).
    If the data is multivariate, Naive Bayes is learned.

    :param data: The data.
    :param distributions: The distributions of the random variables.
    :param domains: The domains of the random variables.
    :param scope: The scope of the leaf.
    :param alpha: Laplace smoothing factor.
    :return: A leaf distribution.
    """
    _, n_features = data.shape
    if n_features == 1:
        sc = scope[0]
        dist = distributions[sc]
        dom = domains[sc]
        leaf = dist(sc)
        leaf.fit(data, dom, alpha=alpha)
        return leaf
    return learn_naive_bayes(data, distributions, domains, scope, learn_mle, alpha=alpha)


def learn_isotonic(data, distributions, domains, scope):
    """
    Learn a leaf using Isotonic method.
    If the data is multivariate, Naive Bayes is learned.

    :param data: The data.
    :param distributions: The distribution of the random variables.
    :param domains: The domain of the random variables.
    :param scope: The scope of the leaf.
    :return: A leaf distribution.
    """
    _, n_features = data.shape
    if n_features == 1:
        sc = scope[0]
        dist = distributions[sc]
        dom = domains[sc]
        continuous = dist.LEAF_TYPE == LeafType.CONTINUOUS
        leaf = Isotonic(sc, continuous=continuous)
        leaf.fit(data, dom)
        return leaf
    return learn_naive_bayes(data, distributions, domains, scope, learn_isotonic)


def learn_cltree(data, distributions, domains, scope, alpha=0.1):
    """
    Learn a leaf using Chow-Liu tree (CLT).
    If the data is univariate, a Maximum Likelihood Estimate leaf is returned.

    :param data: The data.
    :param distributions: The distributions of the random variables.
    :param domains: The domains of the random variables.
    :param scope: The scope of the leaf.
    :param alpha: Laplace smoothing factor.
    :return: A leaf distribution.
    """
    _, n_features = data.shape
    if n_features == 1:  # If univariate, learn using MLE instead
        return learn_mle(data, distributions, domains, scope, alpha=alpha)

    # If multivariate, learn a CLTree
    dists = [distributions[sc] for sc in scope]
    doms = [domains[sc] for sc in scope]
    assert all([d == Bernoulli for d in dists]), "Chow-Liu trees are only available for binary data"
    leaf = BinaryCLTree(scope)
    leaf.fit(data, doms, alpha=alpha)
    return leaf


def learn_naive_bayes(
        data,
        distributions,
        domains,
        scope,
        learn_leaf_func=learn_mle,
        idx_features=None,
        **learn_leaf_params
):
    """
    Learn a leaf as a Naive Bayes model.

    :param data: The data.
    :param distributions: The distribution of the random variables.
    :param domains: The domain of the random variables.
    :param scope: The scope of the leaf.
    :param idx_features: The indices of the features to use. If None, all the features are used.
    :param learn_leaf_func: The function to use to learn the sub-distributions parameters.
    :param learn_leaf_params: Additional parameters for learn_leaf_func.
    :return: A Naive Bayes model distribution.
    """
    if idx_features is None:
        _, n_features = data.shape
        idx_features = range(n_features)

    node = Mul([], scope)
    for i in idx_features:
        sc = scope[i]
        univariate_data = np.expand_dims(data[:, i], axis=-1)
        leaf = learn_leaf_func(univariate_data, distributions, domains, [sc], **learn_leaf_params)
        node.children.append(leaf)
    return node
