from generalized_photometric_neural_network_experiments.dataset.variable_star.variable_star_light_curve_collections import \
    RrLyraeFfiLightCurveCollection, NonRrLyraeFfiLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_eclipsing_binary_light_curve_collection import \
    TessFfiAntiEclipsingBinaryForTransitLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_databases import TessFfiDatabase
from ramjet.photometric_database.derived.tess_ffi_transit_light_curve_collections import \
    TessFfiConfirmedTransitLightCurveCollection, TessFfiNonTransitLightCurveCollection
from ramjet.photometric_database.standard_and_injected_light_curve_database import StandardAndInjectedLightCurveDatabase, \
    OutOfBoundsInjectionHandlingMethod

magnitude_range = (0, 15)


class RrLyraeFfiDatabase(TessFfiDatabase):
    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections = [
            RrLyraeFfiLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            NonRrLyraeFfiLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range),
            TessFfiLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.validation_standard_light_curve_collections = [
            RrLyraeFfiLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            NonRrLyraeFfiLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range),
            TessFfiLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range)
        ]
        self.inference_light_curve_collections = [
            TessFfiLightCurveCollection(magnitude_range=magnitude_range)]
