import os
import time
import json
import argparse
import numpy as np

from deeprob.spn.models.ratspn import GaussianRatSpn, BernoulliRatSpn

from experiments.datasets import load_binary_dataset, load_continuous_dataset, load_vision_dataset
from experiments.datasets import BINARY_DATASETS, CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_samples
from experiments.utils import save_grid_images


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments'
    )
    parser.add_argument(
        'dataset', choices=BINARY_DATASETS + CONTINUOUS_DATASETS + VISION_DATASETS,
        help='The dataset used in the experiment.'
    )
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--rg-depth', type=int, default=1, help='The region graph\'s depth.')
    parser.add_argument('--rg-repetitions', type=int, default=4, help='The region graph\'s number of repetitions.')
    parser.add_argument('--rg-batch', type=int, default=8, help='The region graph\'s number of distribution batches.')
    parser.add_argument('--rg-sum', type=int, default=8, help='The region graph\'s number of sum nodes per region.')
    parser.add_argument(
        '--uniform-loc', nargs=2, type=float, default=None,
        help='Use uniform location for input distributions layer initialization.'
    )
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale',
        action='store_false', help='Whether to optimize scale in Gaussian layers.'
    )
    parser.add_argument('--in-dropout', type=float, default=None, help='The input distributions layer dropout to use.')
    parser.add_argument('--sum-dropout', type=float, default=None, help='The sum layer dropout to use.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()

    is_vision_dataset = args.dataset in VISION_DATASETS
    is_binary_dataset = args.dataset in BINARY_DATASETS
    is_continuous_dataset = args.dataset in CONTINUOUS_DATASETS
    assert is_vision_dataset or args.discriminative is False, \
        'Discriminative setting is not supported for dataset ' + args.dataset

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    transform = None
    if is_binary_dataset:
        data_train, data_valid, data_test = load_binary_dataset('datasets', args.dataset)
    elif is_continuous_dataset:
        data_train, data_valid, data_test = load_continuous_dataset('datasets', args.dataset)
    else:
        if args.discriminative:
            data_train, data_valid, data_test = load_vision_dataset(
                'datasets', args.dataset, unsupervised=False, preproc='standardize'
            )
        else:
            data_train, data_valid, data_test = load_vision_dataset(
                'datasets', args.dataset, unsupervised=True, preproc='standardize'
            )
    n_features = data_train.features_size()
    out_classes = data_train.num_classes() if args.discriminative else 1

    # Create the results directory
    directory = 'ratspn'
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.discriminative:
        directory = os.path.join(directory, 'discriminative')
        os.makedirs(directory, exist_ok=True)
    else:
        directory = os.path.join(directory, 'generative')
        os.makedirs(directory, exist_ok=True)
        directory = os.path.join(directory, args.dataset)
        samples_directory = os.path.join(directory, 'samples')
        os.makedirs(samples_directory, exist_ok=True)

    # Open the results JSON of the chosen dataset
    filepath = os.path.join(directory, args.dataset + '.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            results = json.load(file)
    else:
        results = dict()

    # Build the model
    rg_depth = min(args.rg_depth, int(np.log2(n_features)))

    if is_binary_dataset:
        model = BernoulliRatSpn(
            n_features,
            out_classes=out_classes,
            rg_depth=rg_depth,
            rg_repetitions=args.rg_repetitions,
            n_batch=args.rg_batch,
            n_sum=args.rg_sum,
            in_dropout=args.in_dropout,
            sum_dropout=args.sum_dropout,
            rand_state=rand_state
        )
    else:
        model = GaussianRatSpn(
            n_features,
            out_classes=out_classes,
            rg_depth=rg_depth,
            rg_repetitions=args.rg_repetitions,
            rg_batch=args.rg_batch,
            rg_sum=args.rg_sum,
            rand_state=rand_state,
            uniform_loc=args.uniform_loc,
            optimize_scale=args.optimize_scale
        )

    # Train the model and collect the results
    if args.discriminative:
        nll, metrics = collect_results_discriminative(
            model, data_train, data_valid, data_test,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, optimizer_kwargs={'weight_decay': args.weight_decay}
        )

        results[timestamp] = {
            'nll': nll,
            'metrics': metrics,
            'settings': args.__dict__
        }
        with open(os.path.join(directory, args.dataset + '.json'), 'w') as file:
            json.dump(results, file, indent=4)
    else:
        mean_ll, stddev_ll, bpp = collect_results_generative(
            model, data_train, data_valid, data_test, compute_bpp=is_vision_dataset,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, optimizer_kwargs={'weight_decay': args.weight_decay}
        )

        results[timestamp] = {
            'log_likelihood': {
                'mean': mean_ll,
                'stddev': stddev_ll
            },
            'bpp': bpp,
            'settings': args.__dict__
        }
        with open(os.path.join(directory, args.dataset + '.json'), 'w') as file:
            json.dump(results, file, indent=4)

        if is_vision_dataset:
            n_samples = 10
            images = collect_samples(model, n_samples * n_samples)
            if data_train.transform is not None:
                images = np.asarray([data_train.transform.inverse(s).numpy() for s in images])
            else:
                images = images.numpy()
            images = images.reshape([n_samples, n_samples, *images.shape[1:]])
            images_filename = os.path.join(samples_directory, timestamp + '.png')
            save_grid_images(images, images_filename)
