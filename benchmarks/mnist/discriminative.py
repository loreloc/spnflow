import os
import pandas as pd
from spnflow.wrappers import build_autoregressive_flow_spn
from benchmarks.mnist.utils import *

# The hyper-parameters space.
HYPER_PARAMETERS = [
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 0.8},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 4, 'hidden_units': [64, 64], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 0.8},

    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 0.8},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 5, 'hidden_units': [32, 32], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 0.8},

    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 10, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 10, 'n_repetitions': 20, 'dropout': 0.8},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 1.0},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 20, 'n_repetitions': 10, 'dropout': 0.8},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 1.0},
    {'depth': 6, 'hidden_units': [16, 16], 'n_sum': 20, 'n_repetitions': 20, 'dropout': 0.8},
]

if __name__ == '__main__':
    # Make sure results directory exists
    directory = os.path.join(RESULTS_DIR, "discriminative")
    if not os.path.isdir(directory):
        os.mkdir(directory)

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]

    # Create the hyper-parameters space and results data frame
    csv_cols = list(HYPER_PARAMETERS[0].keys())
    csv_cols.extend(['n_params', 'val_loss', 'val_accuracy'])
    results_df = pd.DataFrame(columns=csv_cols)

    # Get the loss function
    loss_fn = get_loss_function(kind='cross_entropy')

    # Set the evaluation metric
    metrics = ['accuracy']

    for idx, hp in enumerate(HYPER_PARAMETERS):
        # Set some fixed hyper-parameters
        hp['n_batch'] = 4
        hp['regularization'] = 1e-6

        # Build the model
        spn = build_autoregressive_flow_spn(n_features, n_classes, **hp)

        # Compile the model
        spn.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

        # Fit the model
        history = spn.fit(x_train, y_train, beatch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

        # Make directory
        model_path = os.path.join(directory, str(idx).zfill(4))
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # Save the history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(model_path, 'history.csv'))

        # Save some information about the model
        results_df.loc[idx, hp.keys()] = hp
        results_df.loc[idx, 'n_params'] = spn.count_params()

        # Save the validation loss and accuracy at the end of the training
        results_df.loc[idx, 'val_loss'] = history.history['val_loss'][-1]
        results_df.loc[idx, 'val_accuracy'] = history.history['val_accuracy'][-1]

    # Save the hyper-parameters space and results to file
    results_df.to_csv(os.path.join(directory, 'hpspace.csv'))
