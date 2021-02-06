import os
import json
import itertools
import argparse
import numpy as np

from spnflow.torch.models import DgcSpn
from spnflow.torch.utils import compute_mean_quantiles

from experiments.datasets import DatasetTransform, load_vision_dataset
from experiments.datasets import VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative

# Set the hyper-parameters grid space
HYPERPARAMS = {
    'discriminative': {
        'n_batch': [16, 32],
        'sum_channels': [32, 64, 128]
    },
    'generative': {
        'n_batch': [4, 8, 16],
        'sum_channels': [2, 4],
    }
}
HYPERPARAMS_SPACE = {
    'discriminative': [
        dict(zip(HYPERPARAMS['discriminative'].keys(), x))
        for x in itertools.product(*HYPERPARAMS['discriminative'].values())
    ],
    'generative': [
        dict(zip(HYPERPARAMS['generative'].keys(), x))
        for x in itertools.product(*HYPERPARAMS['generative'].values())
    ]
}

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
    parser.add_argument(
        'dataset', choices=VISION_DATASETS, help='The vision dataset used in the experiment.'
    )
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--dropout', type=float, default=None, help='The dropout to use in case of discriminative.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()

    assert args.dropout is None or args.discriminative is True, \
        'Dropout can only be used in discriminative setting'

    # Load the dataset
    transform = DatasetTransform(dequantize=True, standardize=True, flatten=False)
    if args.discriminative:
        (image_train, label_train), (image_valid, label_valid), (image_test, label_test) = load_vision_dataset(
            'datasets', args.dataset, unsupervised=False
        )
        transform.fit(np.vstack([image_train, image_valid]))
        image_train = transform.forward(image_train)
        image_valid = transform.forward(image_valid)
        image_test = transform.forward(image_test)
        image_size = image_train.shape[1:]
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
        image_size = data_train.shape[1:]
        out_classes = 1

    # Create the results directory
    filepath = 'dgcspn'
    discriminative_filepath = os.path.join(filepath, 'discriminative')
    generative_filepath = os.path.join(filepath, 'generative')
    os.makedirs(filepath, exist_ok=True)
    os.makedirs(discriminative_filepath, exist_ok=True)
    os.makedirs(generative_filepath, exist_ok=True)
    results = {}

    # Run hyper-parameters grid search and collect the results
    hp_space = None
    if args.discriminative:
        hp_space = HYPERPARAMS_SPACE['discriminative']
    else:
        hp_space = HYPERPARAMS_SPACE['generative']
    for idx, hp in enumerate(hp_space):
        # Build the model
        if args.discriminative:
            model = DgcSpn(
                image_size, out_classes,
                n_batch=hp['n_batch'],
                sum_channels=hp['sum_channels'],
                depthwise=True,
                n_pooling=2,
                optimize_scale=False,
                in_dropout=args.dropout,
                prod_dropout=args.dropout,
                uniform_loc=(-1.5, 1.5)
            )
        else:
            quantiles_loc = compute_mean_quantiles(data_train, hp['n_batch'])
            model = DgcSpn(
                image_size, out_classes,
                n_batch=hp['n_batch'],
                sum_channels=hp['sum_channels'],
                depthwise=False,
                n_pooling=0,
                optimize_scale=True,
                in_dropout=None,
                prod_dropout=None,
                quantiles_loc=quantiles_loc
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
            with open(os.path.join(discriminative_filepath, args.dataset + '.json'), 'w') as file:
                json.dump(results, file, indent=4)
        else:
            mean_ll, stddev_ll, bpp = collect_results_generative(
                model, data_train, data_valid, data_test, compute_bpp=True,
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
            with open(os.path.join(generative_filepath, args.dataset + '.json'), 'w') as file:
                json.dump(results, file, indent=4)
