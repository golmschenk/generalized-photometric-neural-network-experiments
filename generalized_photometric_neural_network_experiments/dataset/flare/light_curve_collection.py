"""
Code to represent a collection of light curves.
"""
import shutil

import socket

import re
from filelock import FileLock

from typing import Union, Iterable, Optional, List

import numpy as np
import pandas as pd
from pathlib import Path

from generalized_photometric_neural_network_experiments.dataset.flare.generate_flare_frequency_distribution_synthetic_injectables import \
    injectable_flare_frequency_distribution_metadata_path, InjectableFlareFrequencyDistributionFileColumn, \
    InjectableFlareFrequencyDistributionMetadataColumn, injectable_flare_frequency_distributions_directory
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
        path = self.move_path_to_nvme(path)
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
        return list(self.get_paths_for_label_existence_and_splits(self.is_flaring, self.splits))

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
                slope_exists_for_row = pd.notna(
                    metadata_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE])
                intercept_exists_for_row = pd.notna(
                    metadata_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT])
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

    def move_path_to_nvme(self, path: Path) -> Path:
        match = re.match(r"gpu\d{3}", socket.gethostname())
        if match is not None:
            nvme_path = Path("/lscratch/golmsche").joinpath(path)
            if not nvme_path.exists():
                nvme_path.parent.mkdir(exist_ok=True, parents=True)
                nvme_lock_path = nvme_path.parent.joinpath(nvme_path.name + '.lock')
                lock = FileLock(str(nvme_lock_path))
                with lock.acquire():
                    if not nvme_path.exists():
                        nvme_tmp_path = nvme_path.parent.joinpath(nvme_path.name + '.tmp')
                        shutil.copy(path, nvme_tmp_path)
                        nvme_tmp_path.rename(nvme_path)
            return nvme_path
        else:
            return path


class FlareExperimentUpsideDownLightCurveCollection(FlareExperimentLightCurveCollection):
    def load_times_and_fluxes_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and fluxes from a given light curve path.

        :param path: The path to the light curve file.
        :return: The times and the fluxes of the light curve.
        """
        path = self.move_path_to_nvme(path)
        fluxes, times = self.tess_data_interface.load_fluxes_and_times_from_fits_file(path)
        negative_fluxes = -fluxes
        negative_offset_fluxes = negative_fluxes - np.min(negative_fluxes)
        return times, negative_offset_fluxes

    def load_label_from_path(self, path: Path) -> Union[np.ndarray]:
        """
        Loads the label of an example from a corresponding path.

        :param path: The path to load the label for.
        :return: The label.
        """
        return np.array([np.nan, np.nan], dtype=np.float32)


class InjectableFfdLightCurveCollection(LightCurveCollection):
    def __init__(self, splits: Optional[List[int]] = None):
        super().__init__()
        self.metadata_data_frame = pd.read_csv(injectable_flare_frequency_distribution_metadata_path)
        self.splits: Optional[List[int]] = splits

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        path = self.move_path_to_nvme(path)
        light_curve_data_frame = pd.read_feather(path)
        magnifications = light_curve_data_frame[
            InjectableFlareFrequencyDistributionFileColumn.RELATIVE_AMPLITUDE].values
        times = light_curve_data_frame[InjectableFlareFrequencyDistributionFileColumn.TIME__DAYS].values
        return times, magnifications

    def load_label_from_path(self, path: Path) -> Union[np.ndarray]:
        """
        Loads the label of an example from a corresponding path.

        :param path: The path to load the label for.
        :return: The label.
        """
        metadata_row = self.metadata_data_frame[
            self.metadata_data_frame[InjectableFlareFrequencyDistributionMetadataColumn.FILE_NAME] == path.name].iloc[0]
        slope = metadata_row[InjectableFlareFrequencyDistributionMetadataColumn.SLOPE]
        intercept = metadata_row[InjectableFlareFrequencyDistributionMetadataColumn.INTERCEPT]
        return np.array([slope, intercept], dtype=np.float32)

    def get_paths(self) -> Iterable[Path]:
        """
        Gets the paths for the light curves in the collection.

        :return: An iterable of the light curve paths.
        """
        return list(self.get_paths_for_label_existence_and_splits(self.splits))

    def get_paths_for_label_existence_and_splits(self, splits: Optional[List[int]] = None) -> Iterable[Path]:
        """
        Gets the paths for a given label and splits.

        :return: An iterable of the light curve paths.
        """
        paths = []
        for path in injectable_flare_frequency_distributions_directory.glob('*.feather'):
            metadata_row = self.metadata_data_frame[
                self.metadata_data_frame[InjectableFlareFrequencyDistributionMetadataColumn.FILE_NAME] ==
                str(path.name)].iloc[0]
            if splits is not None:
                if metadata_row[MetadataColumnName.SPLIT] not in splits:
                    continue
            paths.append(path)
        return paths

    def move_path_to_nvme(self, path: Path) -> Path:
        match = re.match(r"gpu\d{3}", socket.gethostname())
        if match is not None:
            nvme_path = Path("/lscratch/golmsche").joinpath(path)
            if not nvme_path.exists():
                nvme_path.parent.mkdir(exist_ok=True, parents=True)
                nvme_lock_path = nvme_path.parent.joinpath(nvme_path.name + '.lock')
                lock = FileLock(str(nvme_lock_path))
                with lock.acquire():
                    if not nvme_path.exists():
                        nvme_tmp_path = nvme_path.parent.joinpath(nvme_path.name + '.tmp')
                        shutil.copy(path, nvme_tmp_path)
                        nvme_tmp_path.rename(nvme_path)
            return nvme_path
        else:
            return path
