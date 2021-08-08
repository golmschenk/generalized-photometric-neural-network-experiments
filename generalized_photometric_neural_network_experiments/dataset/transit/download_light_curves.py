"""
Downloads the light curves for the transit dataset.
"""
import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.transit.names_and_paths import metadata_csv_path, MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_light_curves_for_metadata() -> None:
    """
    Downloads the light curves for the metadata.
    """
    metadata_data_frame = pd.read_csv(metadata_csv_path, index_col=False)
    tess_data_interface = TessDataInterface()
    for row_index, row in list(metadata_data_frame.iterrows())[-1:-100:-1]:
        tess_data_interface.download_two_minute_cadence_lightcurve(tic_id=row[MetadataColumnName.TIC_ID],
                                                                   sector=row[MetadataColumnName.SECTOR],
                                                                   save_directory=light_curve_directory)


if __name__ == '__main__':
    download_light_curves_for_metadata()