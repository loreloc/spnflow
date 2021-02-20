import torch


class RunningAverageMetric:
    """Running (batched) average metric"""
    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0.0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)


def torch_get_activation(activation):
    """
    Get the activation function class by its name.

    :param activation: The activation function's name. It can be: 'relu', 'tanh', 'sigmoid'.
    :return: The activation function class.
    """
    if activation == 'relu':
        return torch.nn.ReLU
    elif activation == 'tanh':
        return torch.nn.Tanh
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid
    else:
        raise NotImplementedError('Unknown activation function name' + activation)
