"""
Downloads the light curves for the transit dataset.
"""
from typing import List

from pathlib import Path

import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import metadata_csv_path, MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_light_curves_for_metadata() -> None:
    """
    Downloads the light curves for the metadata.
    """
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tic_ids = metadata_data_frame[MetadataColumnName.TIC_ID].values
    sectors = metadata_data_frame[MetadataColumnName.SECTOR].values
    download_tess_light_curves_for_tic_ids_and_sectors(tic_ids, sectors)


def download_tess_light_curves_for_tic_ids_and_sectors(tic_ids: List[int], sectors: List[int]) -> None:
    """
    Downloads the TESS 2-minute cadence light curves for the given TIC ID and sector lists.

    :param tic_ids: The TIC IDs to download light curves for.
    :param sectors: The sectors to download light curves for.
    """
    tess_data_interface = TessDataInterface()
    path_data_frame = generate_tic_id_and_sector_mapping_to_light_curve_path_data_frame(light_curve_directory)
    for tic_id, sector in zip(tic_ids, sectors):
        try:  # Check if a path already exists for this TIC ID and sector pair.
            existing_path_row = path_data_frame[(path_data_frame['tic_id'] == tic_id) &
                                                (path_data_frame['sector'] == sector)].iloc[0]
            light_curve_path = existing_path_row['path']
        except IndexError:  # Otherwise, download it.
            light_curve_path = tess_data_interface.download_two_minute_cadence_light_curve(tic_id=tic_id, sector=sector,
                                                                                           save_directory=light_curve_directory)
        tess_data_interface.verify_light_curve(light_curve_path)


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


def generate_tic_id_and_sector_mapping_to_light_curve_path_data_frame(directory: Path) -> pd.DataFrame:
    """
    Generates a data frame with existing paths and their corresponding TIC ID and sectors extracted from the path names.

    :param directory: The directory to search for light curves in.
    :return: The data frame.
    """
    tess_data_interface = TessDataInterface()
    paths = list(directory.glob('*.fits'))
    tic_ids = []
    sectors = []
    for path in paths:
        tic_id, sector = tess_data_interface.get_tic_id_and_sector_from_file_path(path)
        tic_ids.append(tic_id)
        sectors.append(sector)
    return pd.DataFrame({'path': paths, 'tic_id': tic_ids, 'sector': sectors})


if __name__ == '__main__':
    download_light_curves_for_metadata()
