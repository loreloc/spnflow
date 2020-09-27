import argparse
import numpy as np

from spnflow.torch.models import RatSpn

from experiments.datasets import load_dataset, load_vision_dataset
from experiments.datasets import get_vision_dataset_inverse_transform
from experiments.datasets import get_vision_dataset_n_classes, get_vision_dataset_n_features
from experiments.utils import collect_results_generative, collect_results_discriminative
from experiments.utils import collect_samples, collect_completions

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Randomized And Tensorized Sum-Product Networks (RAT-SPNs) experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300', 'mnist', 'cifar10'],
        help='The dataset used in the experiment.'
    )
    parser.add_argument(
        '--no-standardize', dest='standardize', action='store_false', help='Whether to disable dataset standardization'
    )
    parser.add_argument('--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--rg-depth', type=int, default=1, help='The region graph depth.')
    parser.add_argument('--rg-repetitions', type=int, default=16, help='The region graph repetitions.')
    parser.add_argument('--n-batch', type=int, default=8, help='The number of batch distributions at leaves.')
    parser.add_argument('--n-sum', type=int, default=8, help='The number of sum nodes per region.')
    parser.add_argument('--dropout', type=float, default=None, help='The dropout layer at sum layers.')
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale', action='store_false',
        help='Whether to disable scale parameters optimization of distribution leaves.'
    )
    parser.add_argument('--n-samples', type=int, default=0, help='The number of samples to store.')
    parser.add_argument('--n-completions', type=int, default=0, help='The number of samples per completion kind.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    args = parser.parse_args()
    settings = vars(args)

    # Check the arguments
    vision_dataset = args.dataset in ['mnist', 'cifar10']
    assert vision_dataset or args.discriminative is False, \
        'Discriminative setting is not supported for dataset \'%s\'' % args.dataset
    assert not args.discriminative or args.n_samples == 0, \
        'Sampling is available only in generative setting for dataset \'%s\'' % args.dataset

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    if vision_dataset:
        n_features = get_vision_dataset_n_features(args.dataset)
        data_train, data_val, data_test = load_vision_dataset(
            'datasets', args.dataset, args.discriminative, standardize=args.standardize, flatten=True
        )
        out_classes = 1 if not args.discriminative else get_vision_dataset_n_classes(args.dataset)
        inv_transform = get_vision_dataset_inverse_transform(args.dataset, args.standardize)
    else:
        data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state)
        _, n_features = data_train.shape
        out_classes = 1
        inv_transform = None

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
    if not args.discriminative:
        collect_results_generative(
            'ratspn', settings, model, data_train, data_val, data_test, bpp=vision_dataset,
            lr=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
        if args.n_samples > 0:
            collect_samples('ratspn', settings, model, args.n_samples, inv_transform)
        if args.n_completions > 0:
            collect_completions('ratspn', settings, model, data_test, args.n_completions, inv_transform)
    else:
        collect_results_discriminative(
            'ratspn', settings, model, data_train, data_val, data_test,
            lr=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
