"""
Names and paths for the flare dataset.
"""

from pathlib import Path

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

flare_data_directory = Path('dataset/flare')
metadata_csv_path = flare_data_directory.joinpath('metadata.csv')
light_curve_directory = flare_data_directory.joinpath('light_curves')


class MetadataColumnName(StrEnum):
    """
    An enum of the flare metadata column names.
    """
    TIC_ID = 'tic_id'
    SECTOR = 'sector'
    FLARE_FREQUENCY_DISTRIBUTION_SLOPE = 'flare_frequency_distribution_slope'
    FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT = 'flare_frequency_distribution_intercept'
    SPLIT = 'split'
