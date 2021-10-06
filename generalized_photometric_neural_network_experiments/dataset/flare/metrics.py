"""
Code for calculating the metrics for the flare experiment using thresholded differences for unknown values.
"""
from tensorflow.python.keras import backend
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import MeanSquaredError, MeanMetricWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops


class FlareThresholdedCalculator:
    """
    A class for calculating the metrics for the flare experiment using thresholded differences for unknown values.
    """
    precalculated_slope_mean = -0.9315392038600727
    precalculated_slope_standard_deviation = 0.38475335278973216
    precalculated_intercept_mean = 30.09453437876961
    precalculated_intercept_standard_deviation = 12.749510973110558
    precalculated_means_tensor = tf.constant([[precalculated_slope_mean, precalculated_intercept_mean]],
                                             dtype=tf.float32)
    precalculated_standard_deviations_tensor = tf.constant([[precalculated_slope_standard_deviation,
                                                             precalculated_intercept_standard_deviation]],
                                                           dtype=tf.float32)

    def __init__(self, slope_threshold: float = -0.274, intercept_threshold: float = 8.39):
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
        # Prevent the derivative from having a NaN by making sure all values used in the later `where` are safe.
        safe_y_true = tf.where(threshold_condition, self.threshold, y_true)
        absolute_difference = math_ops.abs(y_pred - safe_y_true)
        difference = tf.where(threshold_condition, over_threshold_difference, absolute_difference)
        return difference

    def scaled_thresholded_absolute_difference(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the absolute difference with true NaN values being the difference above a given threshold
        with the differences scaled by the precalculated standard deviations of the entire metadata set.

        :param y_true: The true values, with NaNs being calculated based on the difference above a threshold.
        :param y_pred: The predicted values.
        :return: The differences.
        """
        difference = self.thresholded_absolute_difference(y_true, y_pred)
        scaled_difference = self.scale_based_on_precalculated_flare_metadata(difference)
        return scaled_difference

    def squared_scaled_thresholded_difference(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculates the squared difference with true NaN values being the difference above a given threshold
        with the differences scaled by the precalculated standard deviations of the entire metadata set.

        :param y_true: The true values, with NaNs being calculated based on the difference above a threshold.
        :param y_pred: The predicted values.
        :return: The differences.
        """
        scaled_difference = self.scaled_thresholded_absolute_difference(y_true, y_pred)
        squared_scaled_difference = tf.math.square(scaled_difference)
        return squared_scaled_difference

    @staticmethod
    def normalize_based_on_true(y_true: tf.Tensor, values_to_normalize: tf.Tensor) -> tf.Tensor:
        """
        Normalize the values passed based on the true values (likely a batch). Useful for having a weighted
        training loss based on the scale of the input values.

        :param y_true: The values to normalized by.
        :param values_to_normalize: The values to normalize.
        :return: The normalized values.
        """
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

    def scale_based_on_precalculated_flare_metadata(self, unscaled_values: tf.Tensor) -> tf.Tensor:
        """
        Scale the given set of slope and intercept values based on the precalculated standard deviations
        of the entire metadata set. Does not offset by the mean, and is expected to be used with values where
        the offset does not matter (i.e., differences between true and predicted values).

        :param unscaled_values: The values to scale.
        :return: The scaled values.
        """
        scaled_values = unscaled_values / self.precalculated_standard_deviations_tensor
        return scaled_values

    def normalize_based_on_precalculated_flare_metadata(self, unnormalized_values: tf.Tensor) -> tf.Tensor:
        """
        Normalize the given set of slope and intercept values based on the precalculated means and standard deviations
        of the entire metadata set.

        :param unnormalized_values: The values to normalize.
        :return: The normalized values.
        """
        normalized_values = (unnormalized_values - self.precalculated_means_tensor
                             ) / self.precalculated_standard_deviations_tensor
        return normalized_values

    def unnormalize_based_on_precalculated_flare_metadata(self, normalized_values: tf.Tensor) -> tf.Tensor:
        """
        Unnormalize the given set of normalized slope and intercept values based on the precalculated means and standard
        deviations of the entire metadata set.

        :param normalized_values: The values to unnormalize.
        :return: The unnormalized values.
        """
        unnormalized_values = (normalized_values * self.precalculated_standard_deviations_tensor
                               ) + self.precalculated_means_tensor
        return unnormalized_values


class FlareSquaredThresholdedDifferenceLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='flare_squared_thresholded_difference_loss'):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.squared_scaled_thresholded_difference, name=name, reduction=reduction)


class FlareThresholdedAbsoluteDifferenceMetric(MeanMetricWrapper):
    def __init__(self, name='flare_thresholded_absolute_difference_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.scaled_thresholded_absolute_difference, name=name, dtype=dtype)


class FlareSquaredThresholdedDifferenceMetric(MeanMetricWrapper):
    def __init__(self, name='flare_squared_thresholded_difference_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.squared_scaled_thresholded_difference, name=name, dtype=dtype)
