"""
Downloads the light curves for the transit dataset.
"""
from typing import List

from pathlib import Path

import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.cross_experiment.download_light_curves import \
    download_tess_light_curves_for_tic_ids_and_sectors
from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import metadata_csv_path, MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_light_curves_for_metadata() -> None:
    """
    Downloads the light curves for the metadata.
    """
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tic_ids = metadata_data_frame[MetadataColumnName.TIC_ID].values
    sectors = metadata_data_frame[MetadataColumnName.SECTOR].values
    download_tess_light_curves_for_tic_ids_and_sectors(tic_ids, sectors, light_curve_directory)


def delete_light_curves_not_in_meta_data() -> None:
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tess_data_interface = TessDataInterface()
    paths = list(light_curve_directory.glob('*.fits'))
    for path in paths:
        tic_id, sector = tess_data_interface.get_tic_id_and_sector_from_file_path(path)
        target_metadata = metadata_data_frame[
            (metadata_data_frame[MetadataColumnName.TIC_ID] == tic_id) &
            (metadata_data_frame[MetadataColumnName.SECTOR] == sector)
            ]
        if target_metadata.shape[0] == 0:
            print(f'Deleting {path}')
            path.unlink()


if __name__ == '__main__':
    download_light_curves_for_metadata()
