import os
import json
import itertools
import subprocess
from experiments.datasets import BINARY_DATASETS

# The experiments working directory
DIRECTORY = 'spn'

# The hyper-parameters grid space
HYPER_PARAMETERS = {
    '--learn-leaf': ['mle', 'cltree'],
    '--n-clusters': [2, 4, 8],
    '--min-rows-slice': [128, 256, 512],
    '--gtest-threshold': [5.0, 10.0]
}


def extract_metrics(result):
    """
    Utility used to extract relevant metrics from experiment result.

    :param result: The experiment result.
    :return: A dictionary of metrics.
    """
    # Get the test log-likelihood and the learning time
    mean_ll = result['log_likelihood']['test']['mean']
    stddev_ll = result['log_likelihood']['test']['stddev']
    lt = result['learning_time']

    # Get the statistics of the model
    stats = result['statistics']

    # Obtain and parse the hyper-parameters from settings
    hps = dict()
    settings = result['settings']
    for param in HYPER_PARAMETERS:
        name = param.replace('--', '').replace('-', '_').lower()
        hps[name] = settings[name]

    # Build and return the metrics dictionary
    return {
        'hyper_parameters': hps,
        'test': {
            'mean': mean_ll,
            'stddev': stddev_ll
        },
        'learning_time': lt,
        'statistics': stats
    }


if __name__ == '__main__':
    # Set the experiments results filepath
    filepath = os.path.join(DIRECTORY, 'experiments.json')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            experiments = json.load(file)
    else:
        experiments = dict()

    # The list of hyper-parameters to use
    hyper_parameters = [
        dict(zip(HYPER_PARAMETERS.keys(), x))
        for x in itertools.product(*HYPER_PARAMETERS.values())
    ]

    # Run for each dataset
    for dataset in BINARY_DATASETS:
        # Skip datasets already present in results file
        if dataset in experiments.keys():
            continue

        # Run the experiment for each hyper-parameter combination
        for params in hyper_parameters:
            args = ["{} {}".format(p, v) for (p, v) in params.items()]
            command = 'python spn.py {} {}'.format(dataset, ' '.join(args))
            print('Experiment: {}'.format(command))
            subprocess.run(command, shell=True, text=True)

        # Open the experiment results
        with open(os.path.join(DIRECTORY, dataset + '.json'), 'r') as file:
            results = json.load(file)

        # Obtain the best hyper-parameters settings based on the validation set mean log-likelihood
        best_timestamp = max(results, key=lambda t: results[t]['log_likelihood']['valid']['mean'])

        # Save the dataset experiments metrics to file
        experiments[dataset] = extract_metrics(results[best_timestamp])
        with open(filepath, 'w') as file:
            json.dump(experiments, file, indent=4)
