import math

import pytest

from generalized_photometric_neural_network_experiments.dataset.flare.light_curve_collection import \
    FlareExperimentLightCurveCollection
from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import TransitLabel


class TestLightCurveCollection:
    @pytest.mark.integration
    def test_transit_experiment_light_curve_collections_contain_correct_counts(self):
        flaring_light_curve_count = 783
        non_flaring_light_curve_count = 10000
        assert len(list(FlareExperimentLightCurveCollection().get_paths())
                   ) == flaring_light_curve_count + non_flaring_light_curve_count
        assert len(list(FlareExperimentLightCurveCollection(is_flaring=True).get_paths())
                   ) == flaring_light_curve_count
        assert len(list(FlareExperimentLightCurveCollection(is_flaring=False).get_paths())
                   ) == non_flaring_light_curve_count
        for split in range(10):
            assert len(list(FlareExperimentLightCurveCollection(
                is_flaring=True, splits=[split]).get_paths())
                       ) in [math.ceil(flaring_light_curve_count / 10),
                             math.floor(flaring_light_curve_count / 10)]
            assert len(list(FlareExperimentLightCurveCollection(
                is_flaring=False, splits=[split]).get_paths())
                       ) in [math.ceil(non_flaring_light_curve_count / 10),
                             math.floor(non_flaring_light_curve_count / 10)]
        half_splits = list(range(5))
        assert abs(len(list(FlareExperimentLightCurveCollection(
            is_flaring=True, splits=half_splits).get_paths())
                   ) - flaring_light_curve_count / 2) < 5
        assert abs(len(list(FlareExperimentLightCurveCollection(
            is_flaring=False, splits=half_splits).get_paths())
                   ) - non_flaring_light_curve_count / 2) < 5
