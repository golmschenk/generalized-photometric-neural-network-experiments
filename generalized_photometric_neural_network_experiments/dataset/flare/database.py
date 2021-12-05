from generalized_photometric_neural_network_experiments.dataset.flare.light_curve_collection import \
    FlareExperimentLightCurveCollection, FlareExperimentUpsideDownLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class FlareDatabase(StandardAndInjectedLightCurveDatabase):
    def __init__(self):
        super().__init__()
        self.number_of_auxiliary_values = 1
        self.number_of_label_values = 2
        self.training_standard_light_curve_collections = [
            FlareExperimentLightCurveCollection(is_flaring=True, splits=list(range(8))),
            # FlareExperimentUpsideDownLightCurveCollection(is_flaring=True, splits=list(range(8))),
            # FlareExperimentLightCurveCollection(is_flaring=False, splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            FlareExperimentLightCurveCollection(is_flaring=True, splits=[8]),
        ]
        self.inference_light_curve_collections = [
            FlareExperimentLightCurveCollection(is_flaring=True, splits=[9]),
        ]
