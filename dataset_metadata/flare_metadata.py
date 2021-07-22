import io

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import pandas as pd
from pathlib import Path

import requests
from astropy.io import ascii


class ColumnName:
    """
    An enum of the flare metadata column names.
    """
    TIC_ID = 'tic_id'
    FLARE_FREQUENCY_DISTRIBUTION_SLOPE = 'flare_frequency_distribution_slope'
    FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT = 'flare_frequency_distribution_intercept'


def download_maximilian_gunther_meta_data() -> None:
    """
    Gets the relevant metadata from the flare catalog paper by Maximilian Gunther et al.
    https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a

    :return: The data frame of the TIC IDs and flare statistics.
    """
    # noinspection SpellCheckingInspection
    paper_data_table_url = 'https://cfn-live-content-bucket-iop-org.s3.amazonaws.com/journals/1538-3881/159/2/60/1/' \
                           'ajab5d3at1_mrt.txt?AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1627329721&' \
                           'Signature=y5iIOy9UA9ax0TuPVZNVf8hClhY%3D'
    paper_data_table_response = requests.get(paper_data_table_url)
    paper_data_table = ascii.read(io.BytesIO(paper_data_table_response.content))
    paper_data_frame = paper_data_table.to_pandas()
    non_na_paper_data_frame = paper_data_frame[(~paper_data_frame['alpha-FFD'].isna()) &
                                               (~paper_data_frame['beta-FFD'].isna())]
    non_duplicate_paper_data_frame = non_na_paper_data_frame.drop_duplicates(subset=['TESS'], keep='first')

    metadata_data_frame = pd.DataFrame({
        ColumnName.TIC_ID: non_duplicate_paper_data_frame['TESS'],
        ColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE: non_duplicate_paper_data_frame['alpha-FFD'],
        ColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT: non_duplicate_paper_data_frame['beta-FFD'],
    })
    metadata_data_frame.to_csv('dataset_metadata/flare_metadata.csv', index=False)


if __name__ == '__main__':
    download_maximilian_gunther_meta_data()
