from generalized_photometric_neural_network_experiments.experiment.flare.models import \
    HadesWithFlareInterceptLuminosityAddedNoSigmoid

print('Starting imports...', flush=True)
import sys

from generalized_photometric_neural_network_experiments.dataset.flare.database import FlareDatabase
from generalized_photometric_neural_network_experiments.dataset.flare.metrics import \
    FlareSquaredThresholdedDifferenceLoss, FlareThresholdedAbsoluteDifferenceMetric, \
    FlareSquaredThresholdedDifferenceMetric, FrequenciesSquaredThresholdedDifferenceLoss, \
    FrequenciesThresholdedAbsoluteDifferenceMetric, FrequenciesSquaredThresholdedDifferenceMetric, \
    SquaredScaledSlopeDifferenceForKnownFlaringMetricForFrequencies, \
    SquaredScaledInterceptDifferenceForKnownFlaringMetricForFrequencies, \
    SquaredScaledInterceptDifferenceForKnownFlaringMetric, SquaredScaledSlopeDifferenceForKnownFlaringMetric
from ramjet.models.single_layer_model import SingleLayerModelLinearWithAuxiliary

sys.path.append('/att/gpfsfs/briskfs01/ppl/golmsche/ramjet')
from ramjet.basic_models import SimpleLightCurveCnn, SanityCheckNetwork

from ramjet.models.hades import Hades, HadesWithAuxiliaryNoSigmoid

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from pathlib import Path
from generalized_photometric_neural_network_experiments.dataset.transit.database import TransitDatabase
from ramjet.models.cura import Cura, CuraWithLateAuxiliary, CuraWithLateAuxiliaryNoSigmoid
from ramjet.trial import create_logging_callbacks, create_logging_metrics


def train():
    print('Starting training process...', flush=True)
    database = FlareDatabase()
    model = HadesWithFlareInterceptLuminosityAddedNoSigmoid(database.number_of_label_values)
    trial_name = f'{type(model).__name__}_luminosity_as_linear'
    epochs_to_run = 1000
    logs_directory = Path('logs')
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='flare')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = FlareSquaredThresholdedDifferenceLoss(name='loss')
    metrics = [FlareThresholdedAbsoluteDifferenceMetric(), FlareSquaredThresholdedDifferenceMetric(),
               SquaredScaledSlopeDifferenceForKnownFlaringMetric(),
               SquaredScaledInterceptDifferenceForKnownFlaringMetric()]
    optimizer = tf.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=5000, validation_steps=500)
    print('Training done.', flush=True)


if __name__ == '__main__':
    train()
