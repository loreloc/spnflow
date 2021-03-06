import os
import time
import json
import argparse
import numpy as np

from spnflow.torch.models.dgcspn import DgcSpn
from spnflow.utils.data import DataDequantizer, compute_mean_quantiles

from experiments.datasets import load_vision_dataset
from experiments.datasets import VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
    )
    parser.add_argument(
        'dataset', choices=VISION_DATASETS, help='The vision dataset used in the experiment.'
    )
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--n-batches', type=int, default=8, help='The number of input distribution layer batches.')
    parser.add_argument('--sum-channels', type=int, default=8, help='The number of channels at sum layers.')
    parser.add_argument('--depthwise', action='store_true', help='Whether to use depthwise convolution layers.')
    parser.add_argument('--n-pooling', type=int, default=0, help='The number of initial pooling product layers.')
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale',
        action='store_false', help='Whether to optimize scale in Gaussian layers.'
    )
    parser.add_argument(
        '--quantiles-loc', action='store_true', default=False,
        help='Whether to use mean quantiles for leaves initialization.'
    )
    parser.add_argument(
        '--uniform-loc', nargs=2, type=float, default=None,
        help='Use uniform location for leaves initialization.'
    )
    parser.add_argument('--in-dropout', type=float, default=None, help='The input distributions layer dropout to use.')
    parser.add_argument('--sum-dropout', type=float, default=None, help='The sum layer dropout to use.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    assert args.quantiles_loc is False or args.uniform_loc is None, \
        'Only one between --quantiles-loc and --uniform-loc can be defined'

    # Load the dataset
    transform = DataDequantizer()
    if args.discriminative:
        (data_train, label_train), (data_valid, label_valid), (data_test, label_test) = load_vision_dataset(
            'datasets', args.dataset, unsupervised=False
        )
    else:
        data_train, data_valid, data_test = load_vision_dataset('datasets', args.dataset, unsupervised=True)

    transform.fit(data_train)
    data_train = transform.forward(data_train)
    data_valid = transform.forward(data_valid)
    data_test = transform.forward(data_test)
    image_size = data_train.shape[1:]

    if args.discriminative:
        out_classes = len(np.unique(label_train))
        data_train = list(zip(data_train, label_train))
        data_valid = list(zip(data_valid, label_valid))
        data_test = list(zip(data_test, label_test))
    else:
        out_classes = 1

    # Create the results directory
    directory = 'dgcspn'
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.discriminative:
        directory = os.path.join(directory, 'discriminative')
        os.makedirs(directory, exist_ok=True)
    else:
        directory = os.path.join(directory, 'generative')
        os.makedirs(directory, exist_ok=True)

    # Open the results JSON of the chosen dataset
    filepath = os.path.join(directory, args.dataset + '.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            results = json.load(file)
    else:
        results = dict()

    # Compute mean quantiles, if specified
    if args.quantiles_loc:
        quantiles_loc = compute_mean_quantiles(data_train, args.n_batches)
    else:
        quantiles_loc = None

    # Build the model
    model = DgcSpn(
        image_size,
        logit=True,
        out_classes=out_classes,
        n_batch=args.n_batches,
        sum_channels=args.sum_channels,
        depthwise=args.depthwise,
        n_pooling=args.n_pooling,
        optimize_scale=args.optimize_scale,
        in_dropout=args.in_dropout,
        sum_dropout=args.sum_dropout,
        quantiles_loc=quantiles_loc,
        uniform_loc=args.uniform_loc,
        rand_state=rand_state
    )

    # Train the model and collect the results
    if args.discriminative:
        nll, accuracy = collect_results_discriminative(
            model, data_train, data_valid, data_test,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
        )
        results[timestamp] = {
            'nll': nll,
            'accuracy': accuracy,
            'settings': args.__dict__
        }
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)
    else:
        mean_ll, stddev_ll, bpp = collect_results_generative(
            model, data_train, data_valid, data_test, compute_bpp=True,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
        )
        results[timestamp] = {
            'log_likelihood': {
                'mean': mean_ll,
                'stddev': stddev_ll
            },
            'bpp': bpp,
            'settings': args.__dict__
        }
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)
