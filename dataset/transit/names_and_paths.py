"""
Names and paths for the transit dataset.
"""

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
    PLANET = 'planet'
    NON_PLANET = 'non_planet'
