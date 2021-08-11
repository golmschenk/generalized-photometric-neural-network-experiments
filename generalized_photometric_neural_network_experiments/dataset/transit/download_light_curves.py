"""
Downloads the light curves for the transit dataset.
"""
from pathlib import Path

import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import metadata_csv_path, MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_light_curves_for_metadata() -> None:
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tess_data_interface = TessDataInterface()
    path_data_frame = generate_tic_id_and_sector_mapping_to_light_curve_path_data_frame(light_curve_directory)
    for row_index, row in list(metadata_data_frame.iterrows()):
        tic_id = row[MetadataColumnName.TIC_ID]
        sector = row[MetadataColumnName.SECTOR]
        try:  # Check if a path already exists for this TIC ID and sector pair.
            existing_path_row = path_data_frame[(path_data_frame['tic_id'] == tic_id) &
                                                (path_data_frame['sector'] == sector)].iloc[0]
            light_curve_path = existing_path_row['path']
        except IndexError:  # Otherwise, download it.
            light_curve_path = tess_data_interface.download_two_minute_cadence_lightcurve(
                tic_id=tic_id, sector=sector, save_directory=light_curve_directory)
        tess_data_interface.verify_lightcurve(light_curve_path)


def generate_tic_id_and_sector_mapping_to_light_curve_path_data_frame(directory: Path) -> pd.DataFrame:
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
