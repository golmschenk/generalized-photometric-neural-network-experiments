import os

from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_light_curve_collections import \
    RrLyraeFfiLightCurveCollection, NonRrLyraeFfiLightCurveCollection
from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_metadata_manager import \
    VariableStarMetadata, VariableTypeName
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""Code for inference on the contents of a directory."""

import datetime
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_databases import \
    RrLyraeFfiDatabase, magnitude_range
from ramjet.models.hades import FfiHades
from ramjet.trial import infer, infer_distribution

log_name = 'logs/FfiHades_no_bn_2022_03_03_00_28_19'
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = RrLyraeFfiDatabase()
database.inference_light_curve_collections = [
    NonRrLyraeFfiLightCurveCollection(dataset_splits=[9], magnitude_range=magnitude_range)]
inference_dataset = database.generate_inference_dataset()

print('Loading model...', flush=True)
model = FfiHades()
model.load_weights(str(saved_log_directory.joinpath('best_validation_model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
infer_results_path = saved_log_directory.joinpath(f'not_known_to_be_variable_infer_results_{datetime_string}.csv')
infer_distribution(model, inference_dataset, infer_results_path)
