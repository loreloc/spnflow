import tensorflow as tf


@tf.function
def log_loss(y_true, y_pred):
    """
    Log-loss function.

    :param y_true: Dummy input variable.
    :param y_pred: The log-likelihood to maximize.
    :return: The negative mean of the log-likelihood.
    """
    return tf.math.negative(tf.math.reduce_mean(y_pred))
