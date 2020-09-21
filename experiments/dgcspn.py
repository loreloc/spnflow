import argparse
import numpy as np

from spnflow.torch.models import DgcSpn
from spnflow.torch.utils import compute_mean_quantiles

from experiments.datasets import load_vision_dataset
from experiments.datasets import get_vision_dataset_transforms
from experiments.datasets import get_vision_dataset_n_classes, get_vision_dataset_image_size
from experiments.utils import collect_results_generative, collect_results_discriminative, collect_completions

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs) experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', choices=['mnist', 'cifar10'], help='The vision dataset used in the experiment.'
    )
    parser.add_argument( '--discriminative', action='store_true', help='Whether to use discriminative settings.')
    parser.add_argument('--n-batch', type=int, default=8, help='The number of batch distributions at leaves.')
    parser.add_argument('--sum-channels', type=int, default=2, help='The number of sum channels for each sum layer.')
    parser.add_argument('--depthwise', action='store_true', help='Whether to use depthwise product layers.')
    parser.add_argument('--n-pooling', type=int, default=0, help='The number of initial pooling product layers.')
    parser.add_argument('--dropout', type=float, default=None, help='The dropout layer at sum layers.')
    parser.add_argument(
        '--no-optimize-scale', dest='optimize_scale', action='store_false',
        help='Whether to disable scale parameters optimization of distribution leaves.'
    )
    parser.add_argument(
        '--quantiles-loc', action='store_true',
        help='Whether to use quantiles for leaves distributions location initialization.'
    )
    parser.add_argument(
        '--uniform-loc', type=float, default=None, nargs=2,
        help='The uniform range for leaves distributions location initialization.'
    )
    parser.add_argument('--n-completions', type=int, default=0, help='The number of samples per completion kind.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    args = parser.parse_args()
    settings = vars(args)

    # Check the arguments
    assert not args.discriminative or args.n_completions == 0, \
        'Completion is available only in generative setting for dataset \'%s\'' % args.dataset
    assert not args.uniform_loc or not args.quantiles_loc, \
        'Only one between --uniform-loc and --quantiles-loc can be specified'

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    image_size = get_vision_dataset_image_size(args.dataset)
    data_train, data_val, data_test = load_vision_dataset(
        'datasets', args.dataset, args.discriminative, normalize=True
    )
    out_classes = 1 if not args.discriminative else get_vision_dataset_n_classes(args.dataset)
    _, inv_transform = get_vision_dataset_transforms(args.dataset, normalize=True)

    # Set the location initialization parameter
    uniform_loc = None
    quantiles_loc = None
    if args.uniform_loc:
        uniform_loc = tuple(args.uniform_loc)
    elif args.quantiles_loc:
        quantiles_loc = compute_mean_quantiles(data_train.dataset, args.n_batch)

    # Build the model
    model = DgcSpn(
        image_size, out_classes,
        n_batch=args.n_batch,
        sum_channels=args.sum_channels,
        depthwise=args.depthwise,
        n_pooling=args.n_pooling,
        dropout=args.dropout,
        optimize_scale=args.optimize_scale,
        quantiles_loc=quantiles_loc,
        uniform_loc=uniform_loc,
        rand_state=rand_state
    )

    # Train the model and collect the results
    if not args.discriminative:
        collect_results_generative(
            'dgcspn', settings, model, data_train, data_val, data_test, bpp=True,
            lr=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
        if args.n_completions > 0:
            collect_completions('dgcspn', settings, model, data_test, args.n_completions, inv_transform, rand_state)
    else:
        collect_results_discriminative(
            'dgcspn', settings, model, data_train, data_val, data_test,
            lr=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
        )
