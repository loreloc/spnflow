from spnflow.learning.structure import learn_structure


def learn_classifier(data, distributions, **kwargs):
    return learn_structure(data, distributions, root_split='rows', **kwargs)
