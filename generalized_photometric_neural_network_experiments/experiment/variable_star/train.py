import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for running training."""
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_databases import \
    RrLyraeFfiDatabase
from ramjet.basic_models import SimpleFfiLightCurveCnn
from ramjet.models.cura import Cura
from ramjet.models.hades import Hades, FfiHades
from ramjet.photometric_database.derived.moa_survey_none_single_and_binary_database import \
    MoaSurveyNoneSingleAndBinaryDatabase
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardAndInjectedTransitDatabase
from ramjet.trial import create_logging_metrics, create_logging_callbacks


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    database = RrLyraeFfiDatabase()
    model = FfiHades()
    trial_name = f'{type(model).__name__}_corrected_non_rrl_label'
    epochs_to_run = 1000
    logs_directory = Path('logs')
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='rr_lyrae')
    training_dataset, validation_dataset = database.generate_datasets()
    loss_metric = BinaryCrossentropy(name='Loss')
    metrics = create_logging_metrics()
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss_metric, metrics=metrics)
    model.fit(training_dataset, epochs=epochs_to_run, validation_data=validation_dataset, callbacks=logging_callbacks,
              steps_per_epoch=5000, validation_steps=500)
    print('Training done.', flush=True)


if __name__ == '__main__':
    train()
