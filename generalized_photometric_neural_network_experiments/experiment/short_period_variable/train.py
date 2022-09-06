import os

from generalized_photometric_neural_network_experiments.dataset.short_period_variable.short_period_variable_database import \
    ShortPeriodVariableDatabase
from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_databases import \
    RrLyraeFfiDatabase

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for running training."""

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from pathlib import Path

from ramjet.models.hades import FfiHades
from ramjet.trial import create_logging_metrics, create_logging_callbacks


def train():
    """Runs the training."""
    print('Starting training process...', flush=True)
    database = ShortPeriodVariableDatabase()
    model = FfiHades()
    trial_name = f'{type(model).__name__}'
    epochs_to_run = 1000
    logs_directory = Path('logs')
    logging_callbacks = create_logging_callbacks(logs_directory, trial_name, database,
                                                 wandb_entity='ramjet', wandb_project='short_period')
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
