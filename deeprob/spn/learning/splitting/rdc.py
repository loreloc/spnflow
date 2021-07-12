import warnings
import numpy as np
import scipy.sparse as sparse

from itertools import combinations
from sklearn import cluster, cross_decomposition
from sklearn.exceptions import ConvergenceWarning
from deeprob.spn.structure.leaf import LeafType
from deeprob.spn.utils.data import ecdf_data, ohe_data


def rdc_cols(data, distributions, domains, random_state, d=0.3, k=20, s=1.0 / 6.0, nl=np.sin):
    """
    Split the features using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param d: The threshold value that regulates the independence tests among the features.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non linear function to use.
    :return: A features partitioning.
    """
    n_samples, n_features = data.shape
    rdc_features = rdc_transform(data, distributions, domains, random_state, k, s, nl)
    pairwise_comparisons = list(combinations(range(n_features), 2))

    adj_matrix = np.zeros((n_features, n_features), dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for CCA
        for i, j in pairwise_comparisons:
            rdc = rdc_cca(i, j, rdc_features)
            if rdc > d:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    np.fill_diagonal(adj_matrix, 1)
    adj_matrix = sparse.csr_matrix(adj_matrix)
    _, clusters = sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True)
    return clusters


def rdc_rows(data, distributions, domains, random_state, n=2, k=20, s=1.0 / 6.0, nl=np.sin):
    """
    Split the samples using the RDC (Randomized Dependency Coefficient) method.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param n: The number of clusters for KMeans.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non linear function to use.
    :return: A samples partitioning.
    """
    # Transform the samples by RDC
    rdc_samples = np.concatenate(
        rdc_transform(data, distributions, domains, random_state, k, s, nl), axis=1
    )

    # Apply K-Means to the transformed samples
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)  # Ignore convergence warnings for K-Means
        return cluster.KMeans(n, n_init=5, random_state=random_state).fit_predict(rdc_samples)


def rdc_cca(i, j, features):
    """
    Compute the RDC (Randomized Dependency Coefficient) using CCA (Canonical Correlation Analysis).

    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param features: The list of the features.
    :return: The RDC coefficient (the largest canonical correlation coefficient).
    """
    cca = cross_decomposition.CCA(n_components=1, max_iter=128, tol=1e-3)
    x_cca, y_cca = cca.fit_transform(features[i], features[j])
    if np.std(x_cca) < 1e-15 or np.std(y_cca) < 1e-15:  # avoid NaN correlation coefficients
        return 0.0
    return np.corrcoef(x_cca.T, y_cca.T)[0, 1]


def rdc_transform(data, distributions, domains, random_state, k, s, nl):
    """
    Execute the RDC (Randomized Dependency Coefficient) pipeline on some data.

    :param data: The data.
    :param distributions: The data distributions.
    :param domains: The data domains.
    :param random_state: The random state.
    :param k: The size of the latent space.
    :param s: The standard deviation of the gaussian distribution.
    :param nl: The non-linear function to use.
    :return: The transformed data.
    """
    features = []
    for i, dist in enumerate(distributions):
        if dist.LEAF_TYPE == LeafType.DISCRETE:
            feature_matrix = ohe_data(data[:, i], domains[i])
        elif dist.LEAF_TYPE == LeafType.CONTINUOUS:
            feature_matrix = np.expand_dims(data[:, i], axis=-1)
        else:
            raise NotImplementedError("Unknown distribution type " + dist.LEAF_TYPE)
        x = np.apply_along_axis(ecdf_data, 0, feature_matrix)
        o = np.ones((feature_matrix.shape[0], 1))
        features.append(np.hstack((x, o)))

    samples = []
    for x in features:
        z = random_state.randn(x.shape[1], k)
        y = nl(s / x.shape[1] * np.dot(x, z))
        o = np.ones((y.shape[0], 1))
        samples.append(np.hstack((y, o)))
    return samples
