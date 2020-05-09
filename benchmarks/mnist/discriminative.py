import os
import pandas as pd
from spnflow.model.flow import build_rat_spn_flow
from benchmarks.mnist.utils import RESULTS_DIR, EPOCHS, BATCH_SIZE, load_mnist_dataset, get_loss_function


# The hyper-parameters space.
HYPER_PARAMETERS = [
    {'depth': 2, 'n_batch': 8, 'hidden_units': [256, 256], 'n_sum': 10, 'n_repetitions':  8, 'log_scale': True, 'dropout': 0.2},
    {'depth': 2, 'n_batch': 8, 'hidden_units': [256, 256], 'n_sum': 10, 'n_repetitions': 16, 'log_scale': True, 'dropout': 0.2},

    {'depth': 3, 'n_batch': 4, 'hidden_units': [128, 128], 'n_sum': 10, 'n_repetitions':  8, 'log_scale': True, 'dropout': 0.2},
    {'depth': 3, 'n_batch': 4, 'hidden_units': [128, 128], 'n_sum': 10, 'n_repetitions': 16, 'log_scale': True, 'dropout': 0.2},

    {'depth': 2, 'n_batch': 8, 'hidden_units': [256, 256], 'n_sum': 10, 'n_repetitions':  8, 'log_scale': False, 'dropout': 0.2},
    {'depth': 2, 'n_batch': 8, 'hidden_units': [256, 256], 'n_sum': 10, 'n_repetitions': 16, 'log_scale': False, 'dropout': 0.2},

    {'depth': 3, 'n_batch': 4, 'hidden_units': [128, 128], 'n_sum': 10, 'n_repetitions':  8, 'log_scale': False, 'dropout': 0.2},
    {'depth': 3, 'n_batch': 4, 'hidden_units': [128, 128], 'n_sum': 10, 'n_repetitions': 16, 'log_scale': False, 'dropout': 0.2},
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
        # Build the model
        spn = build_rat_spn_flow(n_features, n_classes, **hp)

        # Compile the model
        spn.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

        # Fit the model
        history = spn.fit(x_train, y_train, beatch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

        # Make directory
        model_path = os.path.join(directory, str(idx).zfill(2))
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
