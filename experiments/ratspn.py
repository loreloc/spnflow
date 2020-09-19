import argparse
import numpy as np

from spnflow.torch.models import RatSpn

from experiments.datasets import load_dataset, load_vision_dataset
from experiments.datasets import get_vision_dataset_transforms
from experiments.datasets import get_vision_dataset_n_classes, get_vision_dataset_n_features
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_samples

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300', 'mnist'],
        help='The dataset used in the experiment.'
    )
    parser.add_argument(
        '--discriminative', action='store_true',
        help='Whether to use discriminative settings. Valid only for discriminative datasets.'
    )
    parser.add_argument(
        '--rg-depth', type=int, default=1,
        help='The region graph depth.'
    )
    parser.add_argument(
        '--rg-repetitions', type=int, default=16,
        help='The region graph repetitions.'
    )
    parser.add_argument(
        '--n-batch', type=int, default=8,
        help='The number of batch distributions at leaves.'
    )
    parser.add_argument(
        '--n-sum', type=int, default=8,
        help='The number of sum nodes per region.'
    )
    parser.add_argument(
        '--dropout', type=float, default=None,
        help='The dropout layer at sum layers.'
    )
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale', action='store_false',
        help='Whether to disable scale parameters optimization of distribution leaves.'
    )
    parser.add_argument(
        '--n-samples', type=int, default=0,
        help='The number of samples to store. If the dataset is composed by images this value is squared.'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='The learning rate.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='The batch size.'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='The number of epochs.'
    )
    parser.add_argument(
        '--patience', type=int, default=30,
        help='The epochs patience used for early stopping.'
    )
    args = parser.parse_args()

    # Check the arguments
    vision_dataset = args.dataset in ['mnist']
    assert vision_dataset or args.discriminative is False, \
        'Discriminative setting is not supported for dataset \'%s\'' % args.dataset
    assert not args.discriminative or args.n_samples == 0, \
        'Sampling is available only in generative setting for dataset \'%s\'' % args.dataset

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    if vision_dataset:
        data_train, data_val, data_test = load_vision_dataset(
            'datasets', args.dataset, args.discriminative, flatten=True
        )
        n_features = get_vision_dataset_n_features(args.dataset)
        out_classes = 1 if not args.discriminative else get_vision_dataset_n_classes(args.dataset)
        _, image_transform = get_vision_dataset_transforms(args.dataset, args.discriminative, flatten=True)
    else:
        data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state)
        _, n_features = data_train.shape
        out_classes = 1
        image_transform = None

    # Build the model
    model = RatSpn(
        n_features, out_classes,
        rg_depth=args.rg_depth,
        rg_repetitions=args.rg_repetitions,
        n_batch=args.n_batch,
        n_sum=args.n_sum,
        dropout=args.dropout,
        optimize_scale=args.optimize_scale,
        rand_state=rand_state
    )

    # Train the model and collect the results
    if args.discriminative:
        collect_results_discriminative(
            'ratspn', vars(args), model,
            data_train, data_val, data_test,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience
        )
    else:
        collect_results_generative(
            'ratspn', vars(args), model,
            data_train, data_val, data_test,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience
        )
        if args.n_samples > 0:
            collect_samples('ratspn', model, args.n_samples, image_transform)
