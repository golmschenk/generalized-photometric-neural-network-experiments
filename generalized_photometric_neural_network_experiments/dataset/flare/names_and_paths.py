"""
Names and paths for the flare dataset.
"""

from pathlib import Path

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

flare_data_directory = Path('data/flare')
flare_data_directory.mkdir(parents=True, exist_ok=True)
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
    EQUIVALENT_DURATION_FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT = \
        'equivalent_duration_flare_frequency_distribution_intercept'
    LUMINOSITY__LOG_10_SOLAR_UNITS = 'luminosity__log_10_solar_units'
    SPLIT = 'split'
