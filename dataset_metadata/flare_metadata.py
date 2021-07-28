from typing import Tuple, List

import numpy as np
import io
import pandas as pd
import requests
from astropy.io import ascii
from pathlib import Path
from bokeh.io import show
from bokeh.models import ColumnDataSource, PrintfTickFormatter, Row
from bokeh.plotting import Figure
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from dataset_metadata.transit_metadata import split_tic_id_and_sector_list_equally
from ramjet.data_interface.tess_data_interface import TessDataInterface

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

metadata_csv_path = Path('dataset_metadata/flare_metadata.csv')
light_curve_directory = Path('dataset_metadata/flare_light_curves')


class MetadataColumnName:
    """
    An enum of the flare metadata column names.
    """
    TIC_ID = 'tic_id'
    SECTOR = 'sector'
    FLARE_FREQUENCY_DISTRIBUTION_SLOPE = 'flare_frequency_distribution_slope'
    FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT = 'flare_frequency_distribution_intercept'
    SPLIT = 'split'


def download_maximilian_gunther_metadata_data_frame() -> pd.DataFrame:
    """
    Gets the relevant metadata from the flare catalog paper by Maximilian Gunther et al.
    https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a

    :return: The data frame of the TIC IDs and flare statistics.
    """
    request_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
                                     '(KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'}  # Prevent bot blocking.
    paper_page_response = requests.get('https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a',
                                       headers=request_headers)
    paper_page_soup = BeautifulSoup(paper_page_response.content, 'html.parser')
    table_data_link_element = paper_page_soup.find(class_='wd-jnl-art-btn-table-data')
    paper_data_table_url = table_data_link_element['href']
    paper_data_table_response = requests.get(paper_data_table_url)
    paper_data_table = ascii.read(io.BytesIO(paper_data_table_response.content))
    paper_data_frame = paper_data_table.to_pandas()
    non_na_paper_data_frame = paper_data_frame[(~paper_data_frame['alpha-FFD'].isna()) &
                                               (~paper_data_frame['beta-FFD'].isna())]
    non_duplicate_paper_data_frame = non_na_paper_data_frame.drop_duplicates(subset=['TESS'], keep='first')

    metadata_data_frame = pd.DataFrame({
        MetadataColumnName.TIC_ID: non_duplicate_paper_data_frame['TESS'],
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE: non_duplicate_paper_data_frame['alpha-FFD'],
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT: non_duplicate_paper_data_frame['beta-FFD'],
    })
    return metadata_data_frame


def download_tess_sector_one_to_three_non_flaring_metadata_data_frame(flaring_metadata_data_frame: pd.DataFrame
                                                                      ) -> pd.DataFrame:
    """
    Given the metadata data frame of the flaring targets, gets the non-flaring targets in the 2-minute cadence
    TESS data for sectors 1-3.

    :param flaring_metadata_data_frame: The flaring target metadata.
    :return: The non-flaring target metadata.
    """
    all_observations = TessDataInterface().get_all_two_minute_single_sector_observations()
    sector_one_to_three_observations = all_observations[all_observations['Sector'] <= 3]
    non_flaring_observations = sector_one_to_three_observations[
        ~sector_one_to_three_observations['TIC ID'].isin(flaring_metadata_data_frame[MetadataColumnName.TIC_ID])]
    non_flaring_tic_ids = non_flaring_observations[MetadataColumnName.TIC_ID].unique()
    non_flaring_data_frame = pd.DataFrame({
        MetadataColumnName.TIC_ID: non_flaring_tic_ids,
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE: pd.NA,
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT: pd.NA,
    })
    return non_flaring_data_frame


def download_flare_metadata_csv() -> None:
    """
    Downloads the metadata for the flare application to a CSV.
    """
    flaring_target_metadata_data_frame = download_maximilian_gunther_metadata_data_frame()
    non_flaring_target_metadata_data_frame = download_tess_sector_one_to_three_non_flaring_metadata_data_frame(
        flaring_target_metadata_data_frame)
    tess_data_interface = TessDataInterface()
    labeled_tuple_list: List[Tuple[int, int, float, float, int]] = []
    flaring_tic_id_and_sector_list: List[Tuple[int, int]] = []
    for tic_id in flaring_target_metadata_data_frame[MetadataColumnName.TIC_ID].values:
        sectors = tess_data_interface.get_sectors_target_appears_in(tic_id)
        for sector in sectors:
            if sector <= 3:
                flaring_tic_id_and_sector_list.append((tic_id, sector))
    flaring_splits = split_tic_id_and_sector_list_equally(flaring_tic_id_and_sector_list, number_of_splits=10)
    for split_index, tic_id_and_sector_split in enumerate(flaring_splits):
        for tic_id, sector in tic_id_and_sector_split:
            labeled_tuple_list.append(
                (tic_id, sector,
                 flaring_target_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                 flaring_target_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT],
                 split_index)
            )
    non_flaring_tic_id_and_sector_list: List[Tuple[int, int]] = []
    for tic_id in non_flaring_target_metadata_data_frame[MetadataColumnName.TIC_ID].values:
        sectors = tess_data_interface.get_sectors_target_appears_in(tic_id)
        for sector in sectors:
            if sector <= 3:
                non_flaring_tic_id_and_sector_list.append((tic_id, sector))
    non_flaring_splits = split_tic_id_and_sector_list_equally(non_flaring_tic_id_and_sector_list, number_of_splits=10)
    for split_index, tic_id_and_sector_split in enumerate(non_flaring_splits):
        for tic_id, sector in tic_id_and_sector_split:
            labeled_tuple_list.append((tic_id, sector, pd.NA, pd.NA, split_index))
    metadata_data_frame = pd.DataFrame(labeled_tuple_list,
                                       columns=[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR,
                                                MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE,
                                                MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT,
                                                MetadataColumnName.SPLIT])
    metadata_data_frame.to_csv(metadata_csv_path, index=False)


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


def show_flare_frequency_distribution_plots() -> None:
    """
    Show some plots about the flare frequency distribution statistics.
    """
    metadata_data_frame = pd.read_csv('dataset_metadata/flare_metadata.csv')
    flaring_metadata_data_frame = metadata_data_frame.dropna()
    flaring_metadata_data_frame['y_intercept'] = flaring_metadata_data_frame[
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
    flaring_metadata_data_frame['x_intercept'] = (
            -flaring_metadata_data_frame['y_intercept'] /
            flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE]
    )
    data_source = ColumnDataSource(flaring_metadata_data_frame)
    ffd_figure = Figure(x_range=(0, flaring_metadata_data_frame['x_intercept'].max()),
                        y_range=(0, flaring_metadata_data_frame['y_intercept'].max()))
    ffd_figure.segment(x0=0, y0='y_intercept', x1='x_intercept', y1=0, source=data_source, color='firebrick',
                       alpha=0.2)
    ffd_figure.yaxis.formatter = PrintfTickFormatter(format='10^%s')
    ffd_figure.xaxis.formatter = PrintfTickFormatter(format='10^%s')

    confidence = 0.68
    half_confidence = confidence / 2
    lower_bound_probability = 0.5 - half_confidence
    upper_bound_probability = 0.5 + half_confidence

    slope_distribution_figure = Figure()
    slope_kernel = gaussian_kde(flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE])
    slope_max = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].max()
    slope_min = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].min()
    slope_difference = slope_max - slope_min
    slope_margin = slope_difference * 0.05
    slope_plotting_positions = np.linspace(slope_min - slope_margin, slope_max + slope_margin, 500)
    slope_pdf = slope_kernel(slope_plotting_positions)
    slope_cdf_values = slope_pdf.cumsum() / slope_pdf.cumsum().max()
    slope_ppf = interp1d(slope_cdf_values, slope_plotting_positions, bounds_error=True)
    slope_lower_bound_value = float(slope_ppf(lower_bound_probability))
    slope_median_value = float(slope_ppf(0.5))
    slope_upper_bound_value = float(slope_ppf(upper_bound_probability))
    slope_deviation_plotting_positions = np.linspace(slope_lower_bound_value, slope_upper_bound_value, 500)
    slope_distribution_figure.varea(x=slope_deviation_plotting_positions, y1=0,
                                    y2=slope_kernel(slope_deviation_plotting_positions),
                                    alpha=0.3)
    slope_distribution_figure.segment(x0=slope_median_value, x1=slope_median_value, y0=0,
                                      y1=slope_kernel(slope_median_value))
    slope_distribution_figure.line(slope_plotting_positions, slope_pdf)

    intercept_distribution_figure = Figure()
    intercept_kernel = gaussian_kde(flaring_metadata_data_frame[
                                        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT])
    intercept_max = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].max()
    intercept_min = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].min()
    intercept_difference = intercept_max - intercept_min
    intercept_margin = intercept_difference * 0.05
    intercept_plotting_positions = np.linspace(intercept_min - intercept_margin, intercept_max + intercept_margin, 500)
    intercept_pdf = intercept_kernel(intercept_plotting_positions)
    intercept_cdf_values = intercept_pdf.cumsum() / intercept_pdf.cumsum().max()
    intercept_ppf = interp1d(intercept_cdf_values, intercept_plotting_positions, bounds_error=True)
    intercept_lower_bound_value = float(intercept_ppf(lower_bound_probability))
    intercept_median_value = float(intercept_ppf(0.5))
    intercept_upper_bound_value = float(intercept_ppf(upper_bound_probability))
    intercept_deviation_plotting_positions = np.linspace(intercept_lower_bound_value, intercept_upper_bound_value, 500)
    intercept_distribution_figure.varea(x=intercept_deviation_plotting_positions, y1=0,
                                        y2=intercept_kernel(intercept_deviation_plotting_positions),
                                        alpha=0.3)
    intercept_distribution_figure.segment(x0=intercept_median_value, x1=intercept_median_value, y0=0,
                                          y1=intercept_kernel(intercept_median_value))
    intercept_distribution_figure.line(intercept_plotting_positions, intercept_pdf)

    row = Row(ffd_figure, slope_distribution_figure, intercept_distribution_figure)
    show(row)


if __name__ == '__main__':
    download_flare_metadata_csv()
