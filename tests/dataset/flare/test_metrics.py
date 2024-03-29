import numpy as np
import tensorflow as tf

from generalized_photometric_neural_network_experiments.dataset.flare.metrics import \
    FlareThresholdedCalculator


class TestFlareThresholdedCalculator:
    def test_absolute_difference_with_unthresholded_values(self):
        y_true = tf.convert_to_tensor([[0, 1], [0, 4]], dtype=tf.float32)
        y_pred = tf.convert_to_tensor([[0, 2], [2, 1]], dtype=tf.float32)
        error_calculator = FlareThresholdedCalculator(value0_threshold=-1, value1_threshold=-1)
        difference = error_calculator.thresholded_absolute_difference(y_true, y_pred)
        expected_difference = np.array([[0, 1], [2, 3]])
        assert np.allclose(difference, expected_difference)

    def test_absolute_difference_with_thresholded_values(self):
        y_true = tf.convert_to_tensor([[5, 5], [np.nan, np.nan], [np.nan, np.nan]], dtype=tf.float32)
        y_pred = tf.convert_to_tensor([[3, 4], [2, 2], [4, 4]], dtype=tf.float32)
        error_calculator = FlareThresholdedCalculator(value0_threshold=2, value1_threshold=3)
        difference = error_calculator.thresholded_absolute_difference(y_true, y_pred)
        expected_difference = np.array([[2, 1], [0, 0], [2, 1]])
        assert np.allclose(difference, expected_difference)

    def test_normalize_based_on_true(self):
        y_true = tf.convert_to_tensor([[1, 2], [3, 6], [np.nan, np.nan]], dtype=tf.float32)
        values_to_normalize = tf.convert_to_tensor([[0, 0]], dtype=tf.float32)
        error_calculator = FlareThresholdedCalculator()
        normalized_values = error_calculator.normalize_based_on_true(y_true, values_to_normalize)
        expected_normalized_values = np.array([[-2 / 1, -4 / 2]])
        assert np.allclose(normalized_values, expected_normalized_values)
