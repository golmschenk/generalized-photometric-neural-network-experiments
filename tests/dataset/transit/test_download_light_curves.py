import pandas as pd
import pytest

from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import metadata_csv_path, \
    light_curve_directory


class TestDownloadLightCurves:
    @pytest.mark.integration
    def test_light_curve_count_matches_metadata_count(self):
        metadata_data_frame = pd.read_csv(metadata_csv_path)
        metadata_count = metadata_data_frame.shape[0]
        light_curve_count = len(list(light_curve_directory.glob('*.fits')))
        assert metadata_count == light_curve_count
