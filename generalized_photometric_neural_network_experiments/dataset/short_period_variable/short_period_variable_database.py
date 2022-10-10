from generalized_photometric_neural_network_experiments.dataset.short_period_variable.short_period_variable_light_curve_collection import \
    UniformNoiseLightCurveCollection, MixedSineAndSawtoothWaveLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection
from ramjet.photometric_database.derived.tess_ffi_transit_databases import TessFfiDatabase

magnitude_range = (0, 15)


class ShortPeriodVariableDatabase(TessFfiDatabase):
    def __init__(self):
        super().__init__()
        short_max_period__hours = 5
        short_min_period__hours = 0.25
        short_max_period__days = short_max_period__hours / 24
        short_min_period__days = short_min_period__hours / 24
        short_period_collection = MixedSineAndSawtoothWaveLightCurveCollection(min_period__days=short_min_period__days,
                                                                               max_period__days=short_max_period__days)
        short_period_collection.label = 1
        long_max_period__hours = 20
        long_min_period__hours = 9
        long_max_period__days = long_max_period__hours / 24
        long_min_period__days = long_min_period__hours / 24
        long_period_collection = MixedSineAndSawtoothWaveLightCurveCollection(min_period__days=long_min_period__days,
                                                                              max_period__days=long_max_period__days)
        long_period_collection.label = 0
        noise_collection = UniformNoiseLightCurveCollection()
        noise_collection.label = 0

        self.training_standard_light_curve_collections = [
            TessFfiLightCurveCollection(dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        ]
        self.training_injectee_light_curve_collection = TessFfiLightCurveCollection(
            dataset_splits=list(range(8)), magnitude_range=magnitude_range)
        self.training_injectable_light_curve_collections = [
            short_period_collection,
            long_period_collection,
            noise_collection
        ]
        self.validation_standard_light_curve_collections = [
            TessFfiLightCurveCollection(dataset_splits=[8], magnitude_range=magnitude_range)
        ]
        self.validation_injectee_light_curve_collection = TessFfiLightCurveCollection(
            dataset_splits=[8], magnitude_range=magnitude_range)
        self.validation_injectable_light_curve_collections = [
            short_period_collection,
            long_period_collection,
            noise_collection
        ]
        self.inference_light_curve_collections = [
            TessFfiLightCurveCollection(magnitude_range=magnitude_range)
        ]
