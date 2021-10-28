import tensorflow as tf

from ramjet.models.hades import Hades


class HadesWithFlareInterceptLuminosityAddedNoSigmoid(Hades):
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
