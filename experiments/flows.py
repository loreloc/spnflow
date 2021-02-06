import os
import json
import argparse
import itertools
import numpy as np

from spnflow.torch.models import RealNVP, MAF

from experiments.datasets import DatasetTransform, load_continuous_dataset, load_vision_dataset
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_samples, save_grid_images

# Set the hyper-parameters grid space
HYPERPARAMS = {
    'model': ['nvp', 'maf'],
    'n_flows': [5, 10],
    'depth': [1, 2],
    'units': [128, 512, 1024]
}
HYPERPARAMS_SPACE = [dict(zip(HYPERPARAMS.keys(), x)) for x in itertools.product(*HYPERPARAMS.values())]

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Normalizing Flows experiments')
    parser.add_argument(
        'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS, help='The dataset used in the experiment.'
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
    if is_vision_dataset:
        directory = os.path.join(directory, args.dataset)
        os.makedirs(directory, exist_ok=True)
        samples_directory = os.path.join(directory, 'samples')
        os.makedirs(samples_directory, exist_ok=True)
    results = {}

    # Run hyper-parameters grid search and collect the results
    for idx, hp in enumerate(HYPERPARAMS_SPACE):
        if hp['model'] == 'nvp':
            model = RealNVP(
                n_features,
                n_flows=hp['n_flows'],
                depth=hp['depth'],
                units=hp['units'],
                logit=is_vision_dataset
            )
        elif hp['model'] == 'maf':
            model = MAF(
                n_features,
                n_flows=hp['n_flows'],
                depth=hp['depth'],
                units=hp['units'],
                logit=is_vision_dataset,
                sequential=n_features <= hp['units']
            )
        else:
            raise NotImplementedError("Experiments for model '%s' are not implemented" % hp['model'])

        # Train the model and collect the results
        mean_ll, stddev_ll, bpp = collect_results_generative(
            model, data_train, data_valid, data_test, compute_bpp=is_vision_dataset,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
        )

        # Save the results
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

        if is_vision_dataset:
            n_samples = 8
            samples = collect_samples(model, n_samples * n_samples)
            images = transform.backward(samples)
            images = images.reshape([n_samples, n_samples, *images.shape[1:]])
            images_filename = os.path.join(samples_directory, str(idx) + '.png')
            save_grid_images(images, images_filename)
