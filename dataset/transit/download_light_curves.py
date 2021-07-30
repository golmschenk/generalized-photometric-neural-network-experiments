"""
Downloads the light curves for the transit dataset.
"""
import pandas as pd

from dataset.transit.names_and_paths import metadata_csv_path, MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface


def download_light_curves_for_metadata() -> None:
    """
    Downloads the light curves for the metadata.
    """
    metadata_data_frame = pd.read_csv(metadata_csv_path, index=False)
    tess_data_interface = TessDataInterface()
    for row_index, row in metadata_data_frame.iterrows():
        tess_data_interface.download_two_minute_cadence_lightcurve(tic_id=row[MetadataColumnName.TIC_ID],
                                                                   sector=row[MetadataColumnName.SECTOR],
                                                                   save_directory=light_curve_directory)