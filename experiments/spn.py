import os
import json
import argparse
import numpy as np
from datetime import datetime

from spnflow.structure.leaf import Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.inference import log_likelihood
from spnflow.utils.statistics import get_statistics

from experiments.datasets import load_dataset


def collect_results(settings, spn, data_test, batch_size=1024):
    # Compute the filename string
    filename = 'spn-%s-%s' % (settings['dataset'], datetime.now().strftime('%m%d%H%M'))

    # Compute the log-likelihoods for the test set (batch mode)
    n_samples = len(data_test)
    ll = np.zeros((n_samples, 1))
    for i in range(0, n_samples - batch_size, batch_size):
        ll[i:i+batch_size] = log_likelihood(spn, data_test[i:i+batch_size])
    n_remaining_samples = n_samples % batch_size
    if n_remaining_samples > 0:
        ll[-n_remaining_samples:] = log_likelihood(spn, data_test[-n_remaining_samples:])
    mu_ll = np.mean(ll)
    sigma_ll = np.std(ll) / np.sqrt(n_samples)

    # Save the results to file
    filepath = os.path.join('spn', 'results')
    os.makedirs(filepath, exist_ok=True)
    results = {'log_likelihoods': {'mean': mu_ll, 'stddev': sigma_ll}, 'settings': settings}
    with open(os.path.join(filepath, filename + '.json'), 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Vanilla Sum-Product Networks (SPNs) experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'],
        help='The dataset used in the experiment.'
    )
    parser.add_argument(
        '--no-standardize', dest='standardize', action='store_false', help='Whether to disable dataset standardization'
    )
    parser.add_argument(
        '--learn-leaf', choices=['mle', 'isotonic'], default='mle',
        help='The algorithm used for learning parameters of leaf distributions.'
    )
    parser.add_argument(
        '--split-rows', choices=['kmeans', 'gmm', 'rdc', 'random'], default='rdc',
        help='The algorithm used for rows splitting.'
    )
    parser.add_argument(
        '--split-cols', choices=['rdc', 'random'], default='rdc',
        help='The algorithm used for columns splitting.'
    )
    parser.add_argument(
        '--min-rows-slice', type=int, default=256,
        help='The minimum number of rows slice on a leaf distribution.'
    )
    parser.add_argument(
        '--min-cols-slice', type=int, default=2,
        help='The minimum number of columns slice on a leaf distribution.'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=None,
        help='The number of clusters for rows splitting.'
    )
    args = parser.parse_args()

    # Check the arguments
    rows_clustering = args.split_rows in ['kmeans', 'gmm']
    assert rows_clustering or args.n_clusters is None, \
        'The argument --n-clusters can be specified only if --split-rows is \'kmeans\' or \'gmm\''
    assert not rows_clustering or args.n_clusters is not None, \
        'Please specify --n-clusters argument'
    assert not rows_clustering or args.n_clusters > 1, \
        'The number of clusters must be at least 2'

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state, args.standardize)
    data_train = np.vstack([data_train, data_val])
    rand_state.shuffle(data_train)
    _, n_features = data_train.shape

    # Learn the SPN density estimator
    spn = learn_estimator(
        data_train, [Gaussian] * n_features,
        learn_leaf=args.learn_leaf,
        split_rows=args.split_rows,
        split_cols=args.split_cols,
        min_rows_slice=args.min_rows_slice,
        min_cols_slice=args.min_cols_slice,
        split_rows_kwargs={'n': args.n_clusters} if rows_clustering else {}
    )
    print(get_statistics(spn))

    # Collect the experiments results
    collect_results(vars(args), spn, data_test)
