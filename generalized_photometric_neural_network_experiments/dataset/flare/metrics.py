"""
Code for calculating the metrics for the flare experiment using thresholded differences for unknown values.
"""
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras import backend
import tensorflow as tf
from tensorflow.python.ops import math_ops


class FlareThresholdedCalculator:
    """
    A class for calculating the metrics for the flare experiment using thresholded differences for unknown values.
    """

    def __init__(self, slope_threshold: float = 0, intercept_threshold: float = 0):
        self.slope_threshold = slope_threshold
        self.intercept_threshold = intercept_threshold
        self.threshold = tf.constant([[self.slope_threshold, self.intercept_threshold]], dtype=tf.float32)

    def thresholded_absolute_difference(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the absolute difference with true NaN values being the difference above a given threshold.

        :param y_true: The true values, with NaNs being calculated based on the difference above a threshold.
        :param y_pred: The predicted values.
        :return: The differences.
        """
        over_threshold_difference = backend.maximum(y_pred - self.threshold, tf.constant(0, dtype=tf.float32))
        threshold_condition = math_ops.is_nan(y_true)
        absolute_difference = math_ops.abs(y_pred - y_true)
        difference = tf.where(threshold_condition, over_threshold_difference, absolute_difference)
        return difference

    def normalize_based_on_true(self, y_true: tf.Tensor, values_to_normalize: tf.Tensor) -> tf.Tensor:
        true_slopes = y_true[:, 0]
        true_slope_standard_deviation = tf.math.reduce_std(tf.boolean_mask(true_slopes, tf.math.is_finite(true_slopes)))
        true_slope_mean = tf.math.reduce_mean(tf.boolean_mask(true_slopes, tf.math.is_finite(true_slopes)))
        true_intercepts = y_true[:, 1]
        true_intercept_standard_deviation = tf.math.reduce_std(tf.boolean_mask(true_intercepts,
                                                                               tf.math.is_finite(true_intercepts)))
        true_intercept_mean = tf.math.reduce_mean(tf.boolean_mask(true_intercepts, tf.math.is_finite(true_intercepts)))
        normalized_values = ((values_to_normalize - [[true_slope_mean, true_intercept_mean]]) /
                             [[true_slope_standard_deviation, true_intercept_standard_deviation]])
        return normalized_values
