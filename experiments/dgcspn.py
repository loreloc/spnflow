import os
import json
import itertools
import argparse
import torch
import torchvision

from spnflow.torch.models import DgcSpn
from spnflow.torch.utils import compute_mean_quantiles

from experiments.datasets import load_vision_dataset
from experiments.datasets import get_vision_dataset_inverse_transform
from experiments.datasets import get_vision_dataset_n_classes, get_vision_dataset_image_size
from experiments.datasets import VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_completions

# Set the hyper-parameters grid space
hyperparams = {
    'discriminative': {
        'n_batch': [16, 32],
        'sum_channels': [32, 64, 128]
    },
    'generative': {
        'n_batch': [4, 8, 16],
        'sum_channels': [2, 4],
    }
}
hyperparams_space = {
    'discriminative': [
        dict(zip(hyperparams['discriminative'].keys(), x))
        for x in itertools.product(*hyperparams['discriminative'].values())
    ],
    'generative': [
        dict(zip(hyperparams['generative'].keys(), x))
        for x in itertools.product(*hyperparams['generative'].values())
    ]
}

# Parse the arguments
parser = argparse.ArgumentParser(
    description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments'
)
parser.add_argument(
    'dataset', choices=VISION_DATASETS, help='The vision dataset used in the experiment.'
)
parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
parser.add_argument('--batch-size', type=int, default=128, help='The batch size.')
parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
args = parser.parse_args()

# Load the dataset
image_size = get_vision_dataset_image_size(args.dataset)
data_train, data_val, data_test = load_vision_dataset('datasets', args.dataset, args.discriminative, standardize=True)
out_classes = 1 if not args.discriminative else get_vision_dataset_n_classes(args.dataset)
inv_transform = get_vision_dataset_inverse_transform(args.dataset, standardize=True)

# Create the results directory
filepath = 'dgcspn'
discriminative_filepath = os.path.join(filepath, 'discriminative')
generative_filepath = os.path.join(filepath, 'generative')
samples_filepath = os.path.join(generative_filepath, 'samples')
completions_filepath = os.path.join(generative_filepath, 'completions')
os.makedirs(filepath, exist_ok=True)
os.makedirs(discriminative_filepath, exist_ok=True)
os.makedirs(generative_filepath, exist_ok=True)
os.makedirs(samples_filepath, exist_ok=True)
os.makedirs(completions_filepath, exist_ok=True)
results = {}

# Run hyper-parameters grid search and collect the results
hp_space = hyperparams_space['discriminative'] if args.discriminative else hyperparams_space['generative']
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
            in_dropout=0.2,
            prod_dropout=0.2,
            uniform_loc=(-1.5, 1.5)
        )
    else:
        quantiles_loc = compute_mean_quantiles(data_train.dataset, hp['n_batch'])
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
            model, data_train, data_val, data_test,
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
            model, data_train, data_val, data_test, compute_bpp=True,
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

        # Completions
        n_completions = 16
        samples_full = collect_completions(model, data_test, n_completions, inv_transform)
        filename = os.path.join(completions_filepath, str(idx) + '.png')
        torchvision.utils.save_image(samples_full, filename, nrow=n_completions, padding=0)
