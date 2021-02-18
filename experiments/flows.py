import os
import time
import json
import argparse
import numpy as np

from spnflow.torch.models.flows import RealNVP, MAF

from experiments.datasets import DatasetTransform, load_continuous_dataset, load_vision_dataset
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_samples, save_grid_images


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Normalizing Flows experiments')
    parser.add_argument(
        'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS, help='The dataset used in the experiment.'
    )
    parser.add_argument('model', choices=['nvp', 'maf'], help='The normalizing flow model to use.')
    parser.add_argument('--n-flows', type=int, default=5, help='The number of normalizing flows layers.')
    parser.add_argument(
        '--no-batch-norm', dest='batch_norm',
        action='store_false', help='Whether to use batch normalization.'
    )
    parser.add_argument('--depth', type=int, default=1, help='The depth of each normalizing flow layer.')
    parser.add_argument('--units', type=int, default=128, help='The number of units at each layer.')
    parser.add_argument(
        '--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu', help='The activation function to use.'
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
    is_vision_dataset = args.dataset in VISION_DATASETS
    transform = None
    if is_vision_dataset:
        transform = DatasetTransform(dequantize=True, standardize=False, flatten=True)
        data_train, data_valid, data_test = load_vision_dataset(
            'datasets', args.dataset, unsupervised=True
        )
    else:
        transform = DatasetTransform(standardize=True)
        data_train, data_valid, data_test = load_continuous_dataset('datasets', args.dataset)
    transform.fit(np.vstack([data_train, data_valid]))
    data_train = transform.forward(data_train)
    data_valid = transform.forward(data_valid)
    data_test = transform.forward(data_test)
    _, n_features = data_train.shape

    # Create the results directory
    directory = 'flows'
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if is_vision_dataset:
        directory = os.path.join(directory, args.dataset)
        os.makedirs(directory, exist_ok=True)
        samples_directory = os.path.join(directory, 'samples')
        os.makedirs(samples_directory, exist_ok=True)

    # Open the results JSON of the chosen dataset
    filepath = os.path.join(directory, args.dataset + '.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            results = json.load(file)
    else:
        results = dict()

    if args.model == 'nvp':
        model = RealNVP(
            n_features,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            activation=args.activation,
            logit=is_vision_dataset
        )
    elif args.model == 'maf':
        model = MAF(
            n_features,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            activation=args.activation,
            logit=is_vision_dataset,
            sequential=n_features <= args.units
        )
    else:
        raise NotImplementedError("Experiments for model {} are not implemented".format(args.model))

    # Train the model and collect the results
    mean_ll, stddev_ll, bpp = collect_results_generative(
        model, data_train, data_valid, data_test, compute_bpp=is_vision_dataset,
        lr=args.learning_rate, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
    )

    # Save the results
    results[timestamp] = {
        'log_likelihood': {
            'mean': mean_ll,
            'stddev': 2.0 * stddev_ll
        },
        'bpp': bpp,
        'settings': args.__dict__
    }
    with open(os.path.join(directory, args.dataset + '.json'), 'w') as file:
        json.dump(results, file, indent=4)

    if is_vision_dataset:
        n_samples = 8
        samples = collect_samples(model, n_samples * n_samples)
        images = transform.backward(samples)
        images = images.reshape([n_samples, n_samples, *images.shape[1:]])
        images_filename = os.path.join(samples_directory, str(timestamp) + '.png')
        save_grid_images(images, images_filename)
