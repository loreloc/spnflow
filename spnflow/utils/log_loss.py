import tensorflow as tf


@tf.function
def log_loss(y_true, y_pred):
    return tf.math.negative(tf.math.reduce_mean(y_pred))
