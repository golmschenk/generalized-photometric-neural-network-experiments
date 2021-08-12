"""
Downloads the light curves for the flare dataset.
"""
import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import metadata_csv_path, \
    MetadataColumnName
from generalized_photometric_neural_network_experiments.dataset.transit.download_light_curves import \
    download_tess_light_curves_for_tic_ids_and_sectors


def download_light_curves_for_metadata() -> None:
    """
    Downloads the light curves for the metadata.
    """
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tic_ids = metadata_data_frame[MetadataColumnName.TIC_ID].values
    sectors = metadata_data_frame[MetadataColumnName.SECTOR].values
    download_tess_light_curves_for_tic_ids_and_sectors(tic_ids, sectors)


if __name__ == '__main__':
    download_light_curves_for_metadata()
