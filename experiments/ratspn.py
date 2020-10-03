import os
import json
import math
import itertools
import argparse
import numpy as np
import torchvision

from spnflow.torch.models import RatSpn

from experiments.datasets import load_dataset, load_vision_dataset
from experiments.datasets import get_vision_dataset_inverse_transform
from experiments.datasets import get_vision_dataset_n_classes, get_vision_dataset_n_features
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_results_discriminative
from experiments.utils import collect_image_samples, collect_completions

# Set the hyper-parameters grid space
hyperparams = {
    'rg_depth': [1, 2, 3],
    'rg_repetitions': [8, 16],
    'n_batch': [8, 16]
}
hyperparams_space = [dict(zip(hyperparams.keys(), x)) for x in itertools.product(*hyperparams.values())]

# Parse the arguments
parser = argparse.ArgumentParser(
    description='Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments'
)
parser.add_argument(
    'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS, help='The dataset used in the experiment.'
)
parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
args = parser.parse_args()

# Check the arguments
vision_dataset = args.dataset in VISION_DATASETS
assert vision_dataset or args.discriminative is False, \
    'Discriminative setting is not supported for dataset \'%s\'' % args.dataset

# Instantiate a random state, used for reproducibility
rand_state = np.random.RandomState(42)

# Load the dataset
if vision_dataset:
    n_features = get_vision_dataset_n_features(args.dataset)
    data_train, data_val, data_test = load_vision_dataset(
        'datasets', args.dataset, args.discriminative, standardize=True, flatten=True
    )
    out_classes = 1 if not args.discriminative else get_vision_dataset_n_classes(args.dataset)
    inv_transform = get_vision_dataset_inverse_transform(args.dataset, standardize=True)
else:
    data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state)
    _, n_features = data_train.shape
    out_classes = 1
    inv_transform = None

# Create the results directory
filepath = 'ratspn'
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
for idx, hp in enumerate(hyperparams_space):
    # Build the model
    model = RatSpn(
        n_features, out_classes,
        rg_depth=min(hp['rg_depth'], int(math.log2(n_features))),
        rg_repetitions=hp['rg_repetitions'],
        n_batch=hp['n_batch'],
        n_sum=hp['n_batch'],
        optimize_scale=not args.discriminative,
        in_dropout=0.2 if args.discriminative else None,
        prod_dropout=0.2 if args.discriminative else None,
        rand_state=rand_state
    )

    # Train the model and collect the results
    if args.discriminative:
        nll, accuracy = collect_results_discriminative(
            model, data_train, data_val, data_test,
            lr=args.learning_rate, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, weight_decay=args.weight_decay
        )
        results[str(idx)] = {'nll': nll, 'accuracy': accuracy, 'hyper_params': hp}
        with open(os.path.join(discriminative_filepath, args.dataset + '.json'), 'w') as file:
            json.dump(results, file, indent=4)
    else:
        mean_ll, stddev_ll, bpp = collect_results_generative(
            model, data_train, data_val, data_test, compute_bpp=vision_dataset,
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

        # Image sampling
        if vision_dataset:
            n_samples = 8
            images = collect_image_samples(model, n_samples, inv_transform)
            filename = os.path.join(samples_filepath, str(idx) + '.png')
            torchvision.utils.save_image(images, filename, nrow=n_samples, padding=0)

        # Completions
        if vision_dataset:
            n_completions = 16
            samples_full = collect_completions(model, data_test, n_completions, inv_transform)
            filename = os.path.join(completions_filepath, str(idx) + '.png')
            torchvision.utils.save_image(samples_full, filename, nrow=n_completions, padding=0)
