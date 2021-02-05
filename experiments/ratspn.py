import os
import json
import math
import itertools
import argparse
import numpy as np

from spnflow.torch.models import RatSpn

from experiments.datasets import load_continuous_dataset
from experiments.datasets import CONTINUOUS_DATASETS
from experiments.utils import collect_results_generative

# Set the hyper-parameters grid space
HYPERPARAMS = {
    'rg_depth': [1, 2, 3],
    'rg_repetitions': [8, 16],
    'n_batch': [8, 16]
}
HYPERPARAMS_SPACE = [dict(zip(HYPERPARAMS.keys(), x)) for x in itertools.product(*HYPERPARAMS.values())]


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments'
    )
    parser.add_argument(
        'dataset', choices=CONTINUOUS_DATASETS, help='The dataset used in the experiment.'
    )
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    data_train, data_val, data_test = load_continuous_dataset('datasets', args.dataset, rand_state)
    _, n_features = data_train.shape
    out_classes = 1
    inv_transform = None

    # Create the results directory
    directory = 'ratspn'
    os.makedirs(directory, exist_ok=True)
    results = {}

    # Run hyper-parameters grid search and collect the results
    for idx, hp in enumerate(HYPERPARAMS_SPACE):
        # Build the model
        model = RatSpn(
            n_features, out_classes,
            rg_depth=min(hp['rg_depth'], int(math.log2(n_features))),
            rg_repetitions=hp['rg_repetitions'],
            n_batch=hp['n_batch'],
            n_sum=hp['n_batch'],
            optimize_scale=True,
            in_dropout=None,
            prod_dropout=None,
            rand_state=rand_state
        )

        # Train the model and collect the results
        mean_ll, stddev_ll, bpp = collect_results_generative(
            model, data_train, data_val, data_test, compute_bpp=False,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
        )

        results[str(idx)] = {
            'log_likelihood': {
                'mean': mean_ll,
                'stddev': 2.0 * stddev_ll
            },
            'bpp': bpp,
            'hyper_params': hp
        }
        with open(os.path.join(directory, args.dataset + '.json'), 'w') as file:
            json.dump(results, file, indent=4)
