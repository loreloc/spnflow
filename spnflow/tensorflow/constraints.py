import tensorflow as tf


class Positive(tf.keras.constraints.Constraint):
    """
    Constrains the weights to be positive.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        """
        Initialize the constraint.

        :param epsilon: The epsilon minimum threshold.
        :param kwargs: Other arguments.
        """
        assert epsilon > 0.0
        super(Positive, self).__init__(**kwargs)
        self.epsilon = epsilon

    def __call__(self, w):
        """
        Call the constraint.

        :param w: The weights.
        :return: The constrainted weights.
        """
        return tf.math.maximum(w, self.epsilon)
