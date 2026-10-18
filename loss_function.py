import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        return tf.reduce_mean(weight * tf.pow(1 - y_pred, gamma) * ce)
    return loss
