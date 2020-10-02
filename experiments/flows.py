import argparse
import numpy as np

from spnflow.torch.models import RealNVP, MAF

from experiments.datasets import load_dataset, load_vision_dataset
from experiments.datasets import get_vision_dataset_inverse_transform, get_vision_dataset_n_features
from experiments.datasets import CONTINUOUS_DATASETS, VISION_DATASETS
from experiments.utils import collect_results_generative, collect_samples
from experiments.utils import get_activation_class

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Normalizing Flows experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', choices=CONTINUOUS_DATASETS + VISION_DATASETS,
        help='The dataset used in the experiment.'
    )
    parser.add_argument('model', choices=['nvp', 'maf'], help='The normalizing flow model used in the experiment.')
    parser.add_argument(
        '--no-standardize', dest='standardize', action='store_false', help='Whether to disable dataset standardization'
    )
    parser.add_argument('--n-flows', type=int, default=5, help='The number of stacked normalizing flows layers.')
    parser.add_argument('--depth', type=int, default=1, help='The depth of normalizing flows conditioners.')
    parser.add_argument('--units', type=int, default=128, help='The number of units of each conditioner layer.')
    parser.add_argument(
        '--no-batch-norm', dest='batch_norm', action='store_false', help='Whether to disable batch normalization.'
    )
    parser.add_argument(
        '--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu', help='The activation function to use.'
    )
    parser.add_argument('--random-degrees', action='store_true', help='Whether to use random degrees for MAF models.')
    parser.add_argument('--n-samples', type=int, default=0, help='The number of samples to store.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='The learning rate.')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='The epochs patience used for early stopping.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization factor.')
    args = parser.parse_args()
    settings = vars(args)

    # Check the arguments
    assert args.model == 'maf' or not args.random_degrees, \
        '--random-degrees can only be specified when the model is \'maf\''

    # Instantiate a random state, used for reproducibility
    rand_state = np.random.RandomState(42)

    # Load the dataset
    vision_dataset = args.dataset in ['mnist', 'cifar10']
    if vision_dataset:
        n_features = get_vision_dataset_n_features(args.dataset)
        data_train, data_val, data_test = load_vision_dataset(
            'datasets', args.dataset, dequantize=True, standardize=args.standardize, flatten=True
        )
        inv_transform = get_vision_dataset_inverse_transform(args.dataset)
        apply_logit = not args.standardize
    else:
        data_train, data_val, data_test = load_dataset('datasets', args.dataset, rand_state, args.standardize)
        _, n_features = data_train.shape
        inv_transform = None
        apply_logit = False

    if args.model == 'maf':
        model = MAF(
            n_features,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            activation=get_activation_class(args.activation),
            sequential=not args.random_degrees,
            logit=apply_logit,
            rand_state=rand_state
        )
    elif args.model == 'nvp':
        model = RealNVP(
            n_features,
            n_flows=args.n_flows,
            depth=args.depth,
            units=args.units,
            batch_norm=args.batch_norm,
            activation=get_activation_class(args.activation),
            logit=apply_logit
        )

    # Train the model and collect the results
    collect_results_generative(
        'flows', settings, model, data_train, data_val, data_test, bpp=vision_dataset,
        lr=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        weight_decay=args.weight_decay
    )

    # Sampling
    if args.n_samples > 0:
        collect_samples('flows', settings, model, args.n_samples, inv_transform)
