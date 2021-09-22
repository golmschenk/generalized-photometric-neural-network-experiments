import pandas as pd
from pathlib import Path

from typing import List

from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_tess_light_curves_for_tic_ids_and_sectors(tic_ids: List[int], sectors: List[int],
                                                       light_curve_directory: Path) -> None:
    """
    Downloads the TESS 2-minute cadence light curves for the given TIC ID and sector lists.

    :param tic_ids: The TIC IDs to download light curves for.
    :param sectors: The sectors to download light curves for.
    :param light_curve_directory: Where to download the light curves to.
    """
    tess_data_interface = TessDataInterface()
    path_data_frame = generate_tic_id_and_sector_mapping_to_light_curve_path_data_frame(light_curve_directory)
    for tic_id, sector in zip(tic_ids, sectors):
        try:  # Check if a path already exists for this TIC ID and sector pair.
            existing_path_row = path_data_frame[(path_data_frame['tic_id'] == tic_id) &
                                                (path_data_frame['sector'] == sector)].iloc[0]
            light_curve_path = Path(existing_path_row['path'])
        except IndexError:  # Otherwise, download it.
            light_curve_path = tess_data_interface.download_two_minute_cadence_light_curve(
                tic_id=tic_id, sector=sector, save_directory=light_curve_directory)
        tess_data_interface.verify_light_curve(light_curve_path)


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