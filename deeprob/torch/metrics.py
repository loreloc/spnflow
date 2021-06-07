class RunningAverageMetric:
    """Running (batched) average metric"""
    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def reset(self):
        """
        Reset the running average metric.
        """
        self.metric_accumulator = 0
        self.n_metrics = 0

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)
