import os
import json
import argparse
import itertools
import numpy as np

from spnflow.structure.leaf import Bernoulli, Gaussian
from spnflow.learning.wrappers import learn_estimator
from spnflow.algorithms.inference import log_likelihood
from spnflow.utils.statistics import get_statistics

from experiments.datasets import load_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS

# Set the hyper-parameters grid space
hyperparams = {
    'min_rows_slice': [512, 1024],
    'n_clusters': [2, 4],
    'corr_threshold': [0.2, 0.3, 0.4]
}
hyperparams_space = [dict(zip(hyperparams.keys(), x)) for x in itertools.product(*hyperparams.values())]

# Parse the arguments
parser = argparse.ArgumentParser(description='Vanilla Sum-Product Networks (SPNs) experiments')
parser.add_argument(
    'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS, help='The dataset used in the experiment.'
)
args = parser.parse_args()

# Instantiate a random state, used for reproducibility
rand_state = np.random.RandomState(42)

# Load the dataset
data_train, data_test = load_dataset(
    'datasets', args.dataset, rand_state,
    return_val=False, standardize=args.dataset in CONTINUOUS_DATASETS
)
_, n_features = data_train.shape

# Set the distributions
if args.dataset in BINARY_DATASETS:
    distributions = [Bernoulli] * n_features
else:
    distributions = [Gaussian] * n_features

# Create the results directory
filepath = 'spn'
os.makedirs(filepath, exist_ok=True)
results = {}

# Run hyper-parameters grid search and collect the results
for idx, hp in enumerate(hyperparams_space):
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

    # Compute the log-likelihoods for the test set (batch mode)
    batch_size = 4096
    n_samples = len(data_test)
    ll = np.zeros((n_samples, 1))
    for i in range(0, n_samples - batch_size, batch_size):
        ll[i:i+batch_size] = log_likelihood(spn, data_test[i:i+batch_size])
    n_remaining_samples = n_samples % batch_size
    if n_remaining_samples > 0:
        ll[-n_remaining_samples:] = log_likelihood(spn, data_test[-n_remaining_samples:])

    # Save the results
    results[str(idx)] = {
        'log_likelihood': {
            'mean': np.mean(ll),
            'stddev': 2.0 * np.std(ll) / np.sqrt(n_samples)
        },
        'hyper_params': hp,
        'statistics': get_statistics(spn)
    }
    with open(os.path.join(filepath, args.dataset + '.json'), 'w') as file:
        json.dump(results, file, indent=4)
