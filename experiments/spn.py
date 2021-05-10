import os
import time
import json
import argparse
import numpy as np

from spnflow.structure.leaf import Bernoulli, Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.utils.data import DataStandardizer
from spnflow.utils.statistics import get_statistics

from experiments.datasets import load_binary_dataset, load_continuous_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS
from experiments.utils import evaluate_log_likelihoods


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Vanilla Sum-Product Networks (SPNs) experiments')
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help='The dataset used in the experiment.'
    )
    parser.add_argument(
        '--learn-leaf', choices=['mle', 'isotonic', 'cltree'], default='mle', help='The method for leaf learning.'
    )
    parser.add_argument(
        '--split-rows', choices=['kmeans', 'gmm', 'rdc', 'random'], default='gmm', help='The splitting rows method.'
    )
    parser.add_argument(
        '--split-cols', choices=['gvs', 'rdc', 'random'], default='gvs', help='The splitting columns method.'
    )
    parser.add_argument(
        '--min-rows-slice', type=int, default=256, help='The minimum number of rows for splitting.'
    )
    parser.add_argument(
        '--min-cols-slice', type=int, default=2, help='The minimum number of columns for splitting.'
    )
    parser.add_argument(
        '--n-clusters', type=int, default=2, help='The number of clusters for rows splitting.'
    )
    parser.add_argument(
        '--gtest-threshold', type=float, default=5.0, help='The threshold for the G-Test independence test.'
    )
    parser.add_argument(
        '--rdc-threshold', type=float, default=0.3, help='The threshold for the RDC independence test.'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1, help='Laplace smoothing value.'
    )
    parser.add_argument(
        '--seed', type=int, default=1337, help='The Numpy seed value to use.'
    )
    args = parser.parse_args()

    # Apply the given seed, used for reproducibility
    np.random.seed(args.seed)

    # Load the dataset
    if args.dataset in BINARY_DATASETS:
        data_train, data_valid, data_test = load_binary_dataset('datasets', args.dataset)
    else:
        transform = DataStandardizer()
        data_train, data_valid, data_test = load_continuous_dataset('datasets', args.dataset)
        transform.fit(data_train)
        data_train = transform.forward(data_train)
        data_valid = transform.forward(data_valid)
        data_test = transform.forward(data_test)
    _, n_features = data_train.shape

    # Set the distributions at leaves
    if args.dataset in BINARY_DATASETS:
        distributions = [Bernoulli] * n_features
    else:
        distributions = [Gaussian] * n_features

    # Create the results directory
    directory = 'spn'
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Open the results JSON of the chosen dataset
    filepath = os.path.join(directory, args.dataset + '.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            results = json.load(file)
    else:
        results = dict()

    if args.learn_leaf == 'mle' or args.learn_leaf == 'cltree':
        learn_leaf_params = {'alpha': args.smoothing}
    else:
        learn_leaf_params = dict()

    split_rows_kwargs = dict()
    if args.split_rows == 'kmeans' or args.split_rows == 'gmm':
        split_rows_kwargs['n'] = args.n_clusters
    split_cols_kwargs = dict()
    if args.split_cols == 'gvs':
        split_cols_kwargs['p'] = args.gtest_threshold
    elif args.split_cols == 'rdc':
        split_cols_kwargs['d'] = args.rdc_threshold

    # Learn the SPN density estimator
    start_time = time.perf_counter()
    spn = learn_estimator(
        data=data_train,
        distributions=distributions,
        learn_leaf=args.learn_leaf,
        learn_leaf_params=learn_leaf_params,
        split_rows=args.split_rows,
        split_cols=args.split_cols,
        min_rows_slice=args.min_rows_slice,
        min_cols_slice=args.min_cols_slice,
        split_rows_kwargs=split_rows_kwargs,
        split_cols_kwargs=split_cols_kwargs
    )
    learning_time = time.perf_counter() - start_time

    # Compute the log-likelihoods for the validation and test datasets
    valid_mean_ll, valid_stddev_ll = evaluate_log_likelihoods(spn, data_valid)
    test_mean_ll, test_stddev_ll = evaluate_log_likelihoods(spn, data_test)

    # Save the results
    results[timestamp] = {
        'log_likelihood': {
            'valid': {'mean': valid_mean_ll, 'stddev': valid_stddev_ll},
            'test': {'mean': test_mean_ll, 'stddev': test_stddev_ll}
        },
        'settings': args.__dict__,
        'statistics': get_statistics(spn),
        'learning_time': learning_time
    }
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)
