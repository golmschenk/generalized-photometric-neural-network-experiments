from generalized_photometric_neural_network_experiments.dataset.transit.light_curve_collection import \
    TransitExperimentLightCurveCollection
from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import TransitLabel
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase


class TransitDatabase(StandardAndInjectedLightCurveDatabase):
    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            TransitExperimentLightCurveCollection(label=TransitLabel.PLANET, splits=list(range(8))),
            TransitExperimentLightCurveCollection(label=TransitLabel.NON_PLANET, splits=list(range(8))),
        ]
        self.validation_standard_light_curve_collections = [
            TransitExperimentLightCurveCollection(splits=[8]),
        ]
        self.inference_light_curve_collections = [
            TransitExperimentLightCurveCollection(splits=[9]),
        ]
