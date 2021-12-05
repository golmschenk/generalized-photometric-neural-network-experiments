import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import AveragePooling1D

from ramjet.models.components.light_curve_network_block import LightCurveNetworkBlock
from ramjet.models.cura import Cura, CuraNoSigmoid
from ramjet.models.hades import Hades, HadesNoSigmoid


class HadesWithFlareInterceptLuminosityAddedNoSigmoid(HadesNoSigmoid):
    def __init__(self, number_of_label_values=1):
        super().__init__(number_of_label_values=number_of_label_values)

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = super().call(light_curves, training=training, mask=mask)
        luminosity_shift = tf.concat([tf.zeros_like(auxiliary_informations),
                                      (-x[:, 0:1]) * auxiliary_informations], axis=1)
        outputs = x + luminosity_shift
        return outputs


class CuraWithFlareInterceptLuminosityAddedNoSigmoid(CuraNoSigmoid):
    def __init__(self, number_of_label_values=1):
        super().__init__(number_of_label_values=number_of_label_values)

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = super().call(light_curves, training=training, mask=mask)
        luminosity_shift = tf.concat([tf.zeros_like(auxiliary_informations),
                                      (-x[:, 0:1]) * auxiliary_informations], axis=1)
        outputs = x + luminosity_shift
        return outputs


class FlareNet(Model):
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block3 = LightCurveNetworkBlock(filters=32, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=64, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=10, kernel_size=3, pooling_size=2)
        self.block7 = LightCurveNetworkBlock(filters=10, kernel_size=1, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1)
        self.average_pool = AveragePooling1D(pool_size=123)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = light_curves

        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.average_pool(x, training=training)
        x = self.reshape(x, training=training)

        luminosity_shift = tf.concat([tf.zeros_like(auxiliary_informations),
                                      (-x[:, 0:1]) * auxiliary_informations], axis=1)
        outputs = x + luminosity_shift
        return outputs

class MiniFlareNet(Model):
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2)
        self.block3 = LightCurveNetworkBlock(filters=8, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=16, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=5, kernel_size=3, pooling_size=2)
        self.block7 = LightCurveNetworkBlock(filters=5, kernel_size=1, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1)
        self.average_pool = AveragePooling1D(pool_size=123)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = light_curves

        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.average_pool(x, training=training)
        x = self.reshape(x, training=training)

        luminosity_shift = tf.concat([tf.zeros_like(auxiliary_informations),
                                      (-x[:, 0:1]) * auxiliary_informations], axis=1)
        outputs = x + luminosity_shift
        return outputs

class MicroFlareNet(Model):
    def __init__(self, number_of_label_values=1):
        super().__init__()
        self.block0 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2, batch_normalization=False,
                                             dropout_rate=0)
        self.block1 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block2 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block3 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block4 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block5 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block6 = LightCurveNetworkBlock(filters=4, kernel_size=3, pooling_size=2)
        self.block7 = LightCurveNetworkBlock(filters=4, kernel_size=1, pooling_size=1, batch_normalization=False,
                                             dropout_rate=0)
        self.prediction_layer = Convolution1D(number_of_label_values, kernel_size=1)
        self.average_pool = AveragePooling1D(pool_size=123)
        self.reshape = Reshape([number_of_label_values])

    def call(self, inputs, training=False, mask=None):
        """
        The forward pass of the layer.

        :param inputs: The input tensor.
        :param training: A boolean specifying if the layer should be in training mode.
        :param mask: A mask for the input tensor.
        :return: The output tensor of the layer.
        """
        light_curves, auxiliary_informations = inputs
        x = light_curves

        x = self.block0(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.block7(x, training=training)
        x = self.prediction_layer(x, training=training)
        x = self.average_pool(x, training=training)
        x = self.reshape(x, training=training)

        luminosity_shift = tf.concat([tf.zeros_like(auxiliary_informations),
                                      (-x[:, 0:1]) * auxiliary_informations], axis=1)
        outputs = x + luminosity_shift
        return outputs