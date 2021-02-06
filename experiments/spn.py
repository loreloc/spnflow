import os
import json
import argparse
import itertools
import numpy as np

from spnflow.structure.leaf import Bernoulli, Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.inference import log_likelihood
from spnflow.utils.statistics import get_statistics

from experiments.datasets import DatasetTransform, load_binary_dataset, load_continuous_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS

# Set the hyper-parameters grid space
HYPERPARAMS = {
    'min_rows_slice': [512, 1024],
    'n_clusters': [2, 4],
    'corr_threshold': [0.3, 0.4, 0.5]
}
HYPERPARAMS_SPACE = [dict(zip(HYPERPARAMS.keys(), x)) for x in itertools.product(*HYPERPARAMS.values())]


def evaluate_log_likelihoods(spn, data, batch_size=2048):
    n_samples = len(data)
    ll = np.zeros((n_samples, 1))
    for i in range(0, n_samples - batch_size, batch_size):
        ll[i:i + batch_size] = log_likelihood(spn, data[i:i + batch_size])
    n_remaining_samples = n_samples % batch_size
    if n_remaining_samples > 0:
        ll[-n_remaining_samples:] = log_likelihood(spn, data[-n_remaining_samples:])
    mean_ll = np.mean(ll)
    stddev_ll = 2.0 * np.std(ll) / np.sqrt(n_samples)
    return mean_ll, stddev_ll


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Vanilla Sum-Product Networks (SPNs) experiments')
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help='The dataset used in the experiment.'
    )
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    if args.dataset in BINARY_DATASETS:
        data_train, data_valid, data_test = load_binary_dataset('datasets', args.dataset)
    else:
        transform = DatasetTransform(standardize=True)
        data_train, data_valid, data_test = load_continuous_dataset('datasets', args.dataset)
        transform.fit(np.vstack([data_train, data_valid]))
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
    results = {}

    # Run hyper-parameters grid search and collect the results
    for idx, hp in enumerate(HYPERPARAMS_SPACE):
        # Learn the SPN density estimator
        spn = learn_estimator(
            data=data_train,
            distributions=distributions,
            split_rows='kmeans',
            split_cols='rdc',
            min_rows_slice=hp['min_rows_slice'],
            split_rows_kwargs={'n': hp['n_clusters']},
            split_cols_kwargs={'d': hp['corr_threshold']}
        )

        # Compute the log-likelihoods for the datasets
        train_mean_ll, train_stddev_ll = evaluate_log_likelihoods(spn, data_train)
        valid_mean_ll, valid_stddev_ll = evaluate_log_likelihoods(spn, data_valid)
        test_mean_ll, test_stddev_ll = evaluate_log_likelihoods(spn, data_test)

        # Save the results
        results[str(idx)] = {
            'log_likelihood': {
                'train': {'mean': train_mean_ll, 'stddev': train_stddev_ll},
                'valid': {'mean': valid_mean_ll, 'stddev': valid_stddev_ll},
                'test': {'mean': test_mean_ll, 'stddev': test_stddev_ll}
            },
            'hyper_params': hp,
            'statistics': get_statistics(spn)
        }
        with open(os.path.join(directory, args.dataset + '.json'), 'w') as file:
            json.dump(results, file, indent=4)
