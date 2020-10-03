import os
import json
import argparse
import itertools
import numpy as np
import torch
import torchvision

from spnflow.torch.models import RealNVP, MAF

from experiments.datasets import load_dataset, load_vision_dataset
from experiments.datasets import get_vision_dataset_inverse_transform, get_vision_dataset_n_features
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_image_samples

# Set the hyper-parameters grid space
hyperparams = {
    'model': ['nvp', 'maf'],
    'n_flows': [5, 10],
    'depth': [1, 2],
    'units': [128, 512, 1024]
}
hyperparams_space = [dict(zip(hyperparams.keys(), x)) for x in itertools.product(*hyperparams.values())]

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
vision_dataset = args.dataset in VISION_DATASETS
if vision_dataset:
    n_features = get_vision_dataset_n_features(args.dataset)
    data_train, data_val, data_test = load_vision_dataset('datasets', args.dataset, dequantize=True, flatten=True)
    inv_transform = get_vision_dataset_inverse_transform(args.dataset)
    apply_logit = True
else:
    data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state, standardize=True)
    _, n_features = data_train.shape
    inv_transform = None
    apply_logit = False

# Create the results directory
filepath = 'flows'
samples_filepath = os.path.join(filepath, 'samples')
os.makedirs(filepath, exist_ok=True)
os.makedirs(samples_filepath, exist_ok=True)
results = {}

# Run hyper-parameters grid search and collect the results
for idx, hp in enumerate(hyperparams_space):
    if hp['model'] == 'nvp':
        model = RealNVP(
            n_features,
            n_flows=hp['n_flows'],
            depth=hp['depth'],
            units=hp['units'],
            logit=apply_logit
        )
    elif hp['model'] == 'maf':
        model = MAF(
            n_features,
            n_flows=hp['n_flows'],
            depth=hp['depth'],
            units=hp['units'],
            logit=apply_logit,
            sequential=n_features <= hp['units']
        )
    else:
        raise NotImplementedError("Experiments for model '%s' are not implemented" % hp['model'])

    # Train the model and collect the results
    mean_ll, stddev_ll, bpp = collect_results_generative(
        model, data_train, data_val, data_test, compute_bpp=vision_dataset,
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
    with open(os.path.join(filepath, args.dataset + '.json'), 'w') as file:
        json.dump(results, file, indent=4)

    # Image sampling
    if vision_dataset:
        n_samples = 8
        images = collect_image_samples(model, n_samples, inv_transform)
        filename = os.path.join(samples_filepath, str(idx) + '.png')
        torchvision.utils.save_image(images, filename, nrow=n_samples, padding=0)
