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
    fixed_energies = tf.constant([31, 35], dtype=tf.float32)

    def __init__(self, value0_threshold: float = -0.274, value1_threshold: float = 8.39):
        self.value0_threshold = value0_threshold
        self.value1_threshold = value1_threshold
        self.threshold = tf.constant([[self.value0_threshold, self.value1_threshold]], dtype=tf.float32)

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

    def for_slope_and_intercept_get_frequencies_for_fixed_energies(self, slope: tf.Tensor, intercept: tf.Tensor
                                                                   ) -> tf.Tensor:
        frequencies = self.fixed_energies * slope + intercept
        return frequencies

    def for_frequencies_for_fixed_energies_get_slope_and_intercept(
            self, frequencies: tf.Tensor) -> tf.Tensor:
        slopes = (frequencies[:, 1] - frequencies[:, 0]
                  ) / (self.fixed_energies[1] - self.fixed_energies[0])
        intercepts = frequencies[:, 0] - (slopes + self.fixed_energies[0])
        return tf.stack([slopes, intercepts], axis=1)

    def frequencies_thresholded_absolute_difference(self, y_true_ffd: tf.Tensor,
                                                    y_pred_frequencies: tf.Tensor) -> tf.Tensor:
        y_true_frequencies = self.for_slope_and_intercept_get_frequencies_for_fixed_energies(
            y_true_ffd[:, 0:1], y_true_ffd[:, 1:2])
        difference = self.thresholded_absolute_difference(y_true_frequencies, y_pred_frequencies)
        return difference

    def frequencies_squared_thresholded_difference(self, y_true_ffd: tf.Tensor,
                                                   y_pred_frequencies: tf.Tensor) -> tf.Tensor:
        absolute_difference = self.frequencies_thresholded_absolute_difference(y_true_ffd, y_pred_frequencies)
        difference = tf.math.square(absolute_difference)
        return difference

    def squared_scaled_difference_including_nans(self, y_true_ffd: tf.Tensor, y_pred_ffd: tf.Tensor) -> tf.Tensor:
        absolute_difference = math_ops.abs(y_true_ffd - y_pred_ffd)
        scaled_difference = self.scale_based_on_precalculated_flare_metadata(absolute_difference)
        squared_scaled_difference = tf.math.square(scaled_difference)
        return squared_scaled_difference

    def only_flaring_squared_scaled_difference_slope_mean(self, y_true_ffd: tf.Tensor, y_pred_ffd: tf.Tensor
                                                          ) -> tf.Tensor:
        differences = self.squared_scaled_difference_including_nans(y_true_ffd, y_pred_ffd)
        slope_differences = differences[:, 0]
        slope_mean_difference = tf.experimental.numpy.nanmean(slope_differences)
        return slope_mean_difference

    def only_flaring_squared_scaled_difference_intercept_mean(self, y_true_ffd: tf.Tensor, y_pred_ffd: tf.Tensor
                                                              ) -> tf.Tensor:
        differences = self.squared_scaled_difference_including_nans(y_true_ffd, y_pred_ffd)
        intercept_differences = differences[:, 1]
        intercept_mean_difference = tf.experimental.numpy.nanmean(intercept_differences)
        return intercept_mean_difference

    def only_flaring_squared_scaled_difference_slope_mean_from_frequencies(
            self, y_true_ffd: tf.Tensor, y_pred_frequencies: tf.Tensor) -> tf.Tensor:
        y_pred_ffd = self.for_frequencies_for_fixed_energies_get_slope_and_intercept(y_pred_frequencies)
        differences = self.squared_scaled_difference_including_nans(y_true_ffd, y_pred_ffd)
        slope_differences = differences[:, 0]
        slope_mean_difference = tf.experimental.numpy.nanmean(slope_differences)
        return slope_mean_difference

    def only_flaring_squared_scaled_difference_intercept_mean_from_frequencies(
            self, y_true_ffd: tf.Tensor, y_pred_frequencies: tf.Tensor) -> tf.Tensor:
        y_pred_ffd = self.for_frequencies_for_fixed_energies_get_slope_and_intercept(y_pred_frequencies)
        differences = self.squared_scaled_difference_including_nans(y_true_ffd, y_pred_ffd)
        intercept_differences = differences[:, 1]
        intercept_mean_difference = tf.experimental.numpy.nanmean(intercept_differences)
        return intercept_mean_difference


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


class FrequenciesSquaredThresholdedDifferenceLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='frequencies_squared_thresholded_difference_loss'):
        calculator = FlareThresholdedCalculator(value0_threshold=-0.485, value1_threshold=-9.351)
        super().__init__(calculator.frequencies_squared_thresholded_difference, name=name, reduction=reduction)


class FrequenciesThresholdedAbsoluteDifferenceMetric(MeanMetricWrapper):
    def __init__(self, name='frequencies_thresholded_absolute_difference_metric', dtype=None):
        calculator = FlareThresholdedCalculator(value0_threshold=-0.485, value1_threshold=-9.351)
        super().__init__(calculator.frequencies_thresholded_absolute_difference, name=name, dtype=dtype)


class FrequenciesSquaredThresholdedDifferenceMetric(MeanMetricWrapper):
    def __init__(self, name='frequencies_squared_thresholded_difference_metric', dtype=None):
        calculator = FlareThresholdedCalculator(value0_threshold=-0.485, value1_threshold=-9.351)
        super().__init__(calculator.frequencies_squared_thresholded_difference, name=name, dtype=dtype)


class SquaredScaledSlopeDifferenceForKnownFlaringMetric(MeanMetricWrapper):
    def __init__(self, name='squared_scaled_slope_difference_for_known_flaring_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.only_flaring_squared_scaled_difference_slope_mean, name=name, dtype=dtype)


class SquaredScaledInterceptDifferenceForKnownFlaringMetric(MeanMetricWrapper):
    def __init__(self, name='squared_scaled_intercept_difference_for_known_flaring_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.only_flaring_squared_scaled_difference_intercept_mean, name=name, dtype=dtype)


class SquaredScaledSlopeDifferenceForKnownFlaringMetricForFrequencies(MeanMetricWrapper):
    def __init__(self, name='squared_scaled_slope_difference_for_known_flaring_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.only_flaring_squared_scaled_difference_slope_mean_from_frequencies, name=name,
                         dtype=dtype)


class SquaredScaledInterceptDifferenceForKnownFlaringMetricForFrequencies(MeanMetricWrapper):
    def __init__(self, name='squared_scaled_intercept_difference_for_known_flaring_metric', dtype=None):
        calculator = FlareThresholdedCalculator()
        super().__init__(calculator.only_flaring_squared_scaled_difference_intercept_mean_from_frequencies, name=name,
                         dtype=dtype)
