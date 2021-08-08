import math

import pytest

from generalized_photometric_neural_network_experiments.dataset.transit.light_curve_collection import \
    TransitExperimentLightCurveCollection
from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import TransitLabel


class TestLightCurveCollection:
    @pytest.mark.integration
    def test_transit_experiment_light_curve_collections_contain_correct_counts(self):
        planet_light_curve_count = 552
        non_planet_light_curve_count = 10000
        assert len(list(TransitExperimentLightCurveCollection().get_paths())
                   ) == planet_light_curve_count + non_planet_light_curve_count
        assert len(list(TransitExperimentLightCurveCollection(label=TransitLabel.PLANET).get_paths())
                   ) == planet_light_curve_count
        assert len(list(TransitExperimentLightCurveCollection(label=TransitLabel.NON_PLANET).get_paths())
                   ) == non_planet_light_curve_count
        for split in range(10):
            assert len(list(TransitExperimentLightCurveCollection(
                label=TransitLabel.PLANET, splits=[split]).get_paths())
                       ) in [math.ceil(planet_light_curve_count / 10),
                             math.floor(planet_light_curve_count / 10)]
            assert len(list(TransitExperimentLightCurveCollection(
                label=TransitLabel.NON_PLANET, splits=[split]).get_paths())
                       ) in [math.ceil(non_planet_light_curve_count / 10),
                             math.floor(non_planet_light_curve_count / 10)]
        half_splits = list(range(5))
        assert abs(len(list(TransitExperimentLightCurveCollection(
            label=TransitLabel.PLANET, splits=half_splits).get_paths())
                   ) - planet_light_curve_count / 2) < 5
        assert abs(len(list(TransitExperimentLightCurveCollection(
            label=TransitLabel.NON_PLANET, splits=half_splits).get_paths())
                   ) - non_planet_light_curve_count / 2) < 5
