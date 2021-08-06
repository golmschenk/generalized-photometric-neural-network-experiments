"""
Names and paths for the transit dataset.
"""
from __future__ import annotations

from pathlib import Path

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

transit_data_directory = Path('dataset/transit')
metadata_csv_path = transit_data_directory.joinpath('metadata.csv')
light_curve_directory = transit_data_directory.joinpath('light_curves')


class MetadataColumnName(StrEnum):
    """
    The list of the meta data columns for the transit data set.
    """
    TIC_ID = 'tic_id'
    SECTOR = 'sector'
    LABEL = 'label'
    SPLIT = 'split'


class TransitLabel(StrEnum):
    """
    The list of possible labels for the transit data set.
    """
    NON_PLANET = 'non_planet'
    PLANET = 'planet'

    def to_int(self) -> int:
        """
        Converts the label to a categorical integer representation.

        :return: The float value of the label.
        """
        return list(TransitLabel).index(self)

    @classmethod
    def from_int(cls, integer: int) -> TransitLabel:
        """
        Converts an integer label representation to its corresponding label.

        :param integer: The integer representation of the label.
        :return: The label.
        """
        return list(TransitLabel)[integer]
