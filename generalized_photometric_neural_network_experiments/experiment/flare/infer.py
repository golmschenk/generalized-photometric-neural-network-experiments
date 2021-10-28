import datetime
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.flare.database import FlareDatabase
from ramjet.models.cura import CuraWithLateAuxiliaryNoSigmoid
from ramjet.models.hades import Hades, HadesWithAuxiliaryNoSigmoid
from ramjet.trial import infer

log_name = '/Users/golmsche/Desktop/CuraWithLateAuxiliaryNoSigmoid_2021_10_06_12_34_59'  # Specify the path to the model to use.
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = FlareDatabase()
inference_dataset = database.generate_inference_dataset()

print('Loading model...', flush=True)
model = CuraWithLateAuxiliaryNoSigmoid(database.number_of_label_values)
model.load_weights(str(saved_log_directory.joinpath('latest_model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
infer_results_path = saved_log_directory.joinpath(f'infer_results_{datetime_string}.csv')
infer(model, inference_dataset, infer_results_path)
