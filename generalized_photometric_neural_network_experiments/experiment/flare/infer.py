import datetime
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.flare.database import FlareDatabase
from generalized_photometric_neural_network_experiments.experiment.flare.models import \
    HadesWithFlareInterceptLuminosityAddedNoSigmoid
from ramjet.models.cura import CuraWithLateAuxiliaryNoSigmoid
from ramjet.models.hades import Hades, HadesWithAuxiliaryNoSigmoid
from ramjet.trial import infer

log_name = 'logs/HadesWithFlareInterceptLuminosityAddedNoSigmoid_plain_2021_11_11_14_29_11'
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = FlareDatabase()
inference_dataset = database.generate_inference_dataset()

print('Loading model...', flush=True)
model = HadesWithFlareInterceptLuminosityAddedNoSigmoid(database.number_of_label_values)
model.load_weights(str(saved_log_directory.joinpath('best_validation_model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
infer_results_path = saved_log_directory.joinpath(f'infer_results_{datetime_string}.csv')
infer(model, inference_dataset, infer_results_path)
