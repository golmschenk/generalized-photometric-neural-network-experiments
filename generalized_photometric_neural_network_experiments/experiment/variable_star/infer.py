import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for inference on the contents of a directory."""

import datetime
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_databases import \
    RrLyraeFfiDatabase
from ramjet.models.hades import FfiHades
from ramjet.trial import infer

log_name = 'logs/FfiHades_corrected_non_rrl_label_2021_12_31_14_37_00'
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = RrLyraeFfiDatabase()
inference_dataset = database.generate_inference_dataset()

print('Loading model...', flush=True)
model = FfiHades()
model.load_weights(str(saved_log_directory.joinpath('best_validation_model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
infer_results_path = saved_log_directory.joinpath(f'infer_results_{datetime_string}.csv')
infer(model, inference_dataset, infer_results_path, number_of_top_predictions_to_keep=100_000)