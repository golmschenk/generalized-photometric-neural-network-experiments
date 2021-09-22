"""
Code to represent a collection of light curves.
"""
from typing import Union, Iterable, Optional, List

import numpy as np
import pandas as pd
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import metadata_csv_path, \
    MetadataColumnName, light_curve_directory
from ramjet.data_interface.tess_data_interface import TessDataInterface
from ramjet.photometric_database.light_curve_collection import LightCurveCollection


class FlareExperimentLightCurveCollection(LightCurveCollection):
    """
    A class to represent a collection of light curves related to the flare experiment.
    """
    tess_data_interface = TessDataInterface()

    def __init__(self, is_flaring: Optional[bool] = None, splits: Optional[List[int]] = None):
        super().__init__()
        self.metadata_data_frame = pd.read_csv(metadata_csv_path)
        self.is_flaring: Optional[bool] = is_flaring
        self.splits: Optional[List[int]] = splits

    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        return times, fluxes

    def load_label_from_path(self, path: Path) -> Union[np.ndarray]:
        """
        Loads the label of an example from a corresponding path.

        :param path: The path to load the label for.
        :return: The label.
        """
        tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(path)
        metadata_row = self.get_metadata_row_for_tic_id_and_sector(tic_id, sector)
        slope = metadata_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE]
        intercept = metadata_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
        return np.array([slope, intercept], dtype=np.float32)

    def get_metadata_row_for_tic_id_and_sector(self, tic_id: int, sector: int) -> pd.Series:
        """
        Gets the metadata row for a given TIC ID and sector.

        :param tic_id: The TIC ID to lookup.
        :param sector: The sector to lookup.
        :return: The metadata row.
        """
        metadata_row = self.metadata_data_frame[
            (self.metadata_data_frame[MetadataColumnName.TIC_ID] == tic_id) &
            (self.metadata_data_frame[MetadataColumnName.SECTOR] == sector)
            ].iloc[0]
        return metadata_row

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        return self.get_paths_for_label_existence_and_splits(self.is_flaring, self.splits)

    def get_paths_for_label_existence_and_splits(self, is_flaring: Optional[bool] = None,
                                                 splits: Optional[List[int]] = None) -> Iterable[Path]:
        """
        Gets the paths for a given label and splits.

        :return: An iterable of the light curve paths.
        """
        paths = []
        for fits_path in light_curve_directory.glob('*.fits'):
            tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(fits_path)
            metadata_row = self.get_metadata_row_for_tic_id_and_sector(tic_id, sector)
            if is_flaring is not None:
                slope_exists_for_row = metadata_row[
                    MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].notna()
                intercept_exists_for_row = metadata_row[
                    MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].notna()
                assert slope_exists_for_row == intercept_exists_for_row
                if is_flaring and not slope_exists_for_row:
                    continue
                if not is_flaring and slope_exists_for_row:
                    continue
            if splits is not None:
                if metadata_row[MetadataColumnName.SPLIT] not in splits:
                    continue
            paths.append(fits_path)
        return paths

    def load_auxiliary_information_for_path(self, path: Path) -> np.ndarray:
        """
        Loads auxiliary information information for the given path.

        :param path: The path to the light curve file.
        :return: The auxiliary information.
        """
        tic_id, sector = self.tess_data_interface.get_tic_id_and_sector_from_file_path(path)
        metadata_row = self.get_metadata_row_for_tic_id_and_sector(tic_id, sector)
        luminosity = metadata_row[MetadataColumnName.LUMINOSITY__LOG_10_SOLAR_UNITS]
        return np.array([luminosity], dtype=np.float32)
