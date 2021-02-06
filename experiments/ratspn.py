import os
import json
import math
import itertools
import argparse
import numpy as np

from spnflow.torch.models import RatSpn

from experiments.datasets import DatasetTransform, load_continuous_dataset, load_vision_dataset
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative

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
        'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS, help='The dataset used in the experiment.'
    )
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--dropout', type=float, default=None, help='The dropout to use in case of discriminative.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()

    is_vision_dataset = args.dataset in VISION_DATASETS
    assert is_vision_dataset or args.discriminative is False, \
        'Discriminative setting is not supported for dataset \'%s\'' % args.dataset
    assert args.dropout is None or args.discriminative is True, \
        'Dropout can only be used in discriminative setting'

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    transform = None
    if is_vision_dataset:
        transform = DatasetTransform(dequantize=True, standardize=True, flatten=True)
        if args.discriminative:
            (image_train, label_train), (image_valid, label_valid), (image_test, label_test) = load_vision_dataset(
                'datasets', args.dataset, unsupervised=False
            )
            transform.fit(np.vstack([image_train, image_valid]))
            image_train = transform.forward(image_train)
            image_valid = transform.forward(image_valid)
            image_test = transform.forward(image_test)
            _, n_features = image_train.shape
            out_classes = len(np.unique(label_train))
            data_train = list(zip(image_train, label_train))
            data_valid = list(zip(image_valid, label_valid))
            data_test = list(zip(image_test, label_test))
        else:
            data_train, data_valid, data_test = load_vision_dataset(
                'datasets', args.dataset, unsupervised=True
            )
            transform.fit(np.vstack([data_train, data_valid]))
            data_train = transform.forward(data_train)
            data_valid = transform.forward(data_valid)
            data_test = transform.forward(data_test)
            _, n_features = data_train.shape
            out_classes = 1
    else:
        transform = DatasetTransform(standardize=True)
        data_train, data_valid, data_test = load_continuous_dataset('datasets', args.dataset)
        transform.fit(np.vstack([data_train, data_valid]))
        data_train = transform.forward(data_train)
        data_valid = transform.forward(data_valid)
        data_test = transform.forward(data_test)
        _, n_features = data_train.shape
        out_classes = 1

    # Create the results directory
    directory = 'ratspn'
    discriminative_directory = os.path.join(directory, 'discriminative')
    generative_directory = os.path.join(directory, 'generative')
    os.makedirs(directory, exist_ok=True)
    os.makedirs(discriminative_directory, exist_ok=True)
    os.makedirs(generative_directory, exist_ok=True)
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
            in_dropout=args.dropout,
            prod_dropout=args.dropout,
            rand_state=rand_state
        )

        # Train the model and collect the results
        if args.discriminative:
            nll, accuracy = collect_results_discriminative(
                model, data_train, data_valid, data_test,
                lr=args.learning_rate, batch_size=args.batch_size,
                epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
            )

            results[str(idx)] = {
                'nll': nll,
                'accuracy': accuracy,
                'hyper_params': hp
            }
            with open(os.path.join(discriminative_directory, args.dataset + '.json'), 'w') as file:
                json.dump(results, file, indent=4)
        else:
            mean_ll, stddev_ll, bpp = collect_results_generative(
                model, data_train, data_valid, data_test, compute_bpp=is_vision_dataset,
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
            with open(os.path.join(generative_directory, args.dataset + '.json'), 'w') as file:
                json.dump(results, file, indent=4)