import pandas as pd
import pytest

from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import metadata_csv_path


class TestGenerateMetadata:
    @pytest.mark.integration
    def test_no_duplicate_tic_id_and_sector_rows(self):
        metadata_data_frame = pd.read_csv(metadata_csv_path)
        duplicated_data_frame = metadata_data_frame[metadata_data_frame.duplicated(['tic_id', 'sector'])]
        assert duplicated_data_frame.shape[0] == 0
