from random import Random

from typing import Tuple, List

import numpy as np
import io
import pandas as pd
import requests
from astropy.io import ascii
from bs4 import BeautifulSoup

from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import metadata_csv_path, \
    MetadataColumnName
from generalized_photometric_neural_network_experiments.dataset.transit.generate_metadata import \
    split_tic_id_and_sector_list_equally
from ramjet.data_interface.tess_data_interface import TessDataInterface, ColumnName as TessColumnName


def download_maximilian_gunther_metadata_data_frame() -> pd.DataFrame:
    """
    Gets the relevant metadata from the flare catalog paper by Maximilian Gunther et al.
    https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a

    :return: The data frame of the TIC IDs and flare statistics.
    """
    request_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
                                     '(KHTML, like Gecko) Version/15.1 Safari/605.1.15'}  # Prevent bot blocking.
    paper_page_response = requests.get('https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a',
                                       headers=request_headers)
    paper_page_soup = BeautifulSoup(paper_page_response.content, 'html.parser')
    if 'captcha' in paper_page_soup.find('title').text.lower():
        raise ConnectionError('The journal page (correctly) thinks this is a bot. Manually visit '
                              'https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a then try again.')
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


def download_data_frame_of_tess_non_flaring_metadata_from_sectors_with_flaring_metadata(
        flaring_metadata_data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Given the metadata data frame of the flaring targets, gets the non-flaring targets in the 2-minute cadence
    TESS data for corresponding sectors.

    :param flaring_metadata_data_frame: The flaring target metadata.
    :return: The non-flaring target metadata.
    """
    tess_data_interface = TessDataInterface()
    all_observations = tess_data_interface.get_all_two_minute_single_sector_observations()
    flaring_sectors = []
    for index, row in flaring_metadata_data_frame.iterrows():
        target_sectors = tess_data_interface.get_sectors_target_appears_in(int(row[MetadataColumnName.TIC_ID]))
        flaring_sectors.extend(target_sectors)
    flaring_sectors = [sector for sector in set(flaring_sectors) if sector <= 26]
    sector_one_to_three_observations = all_observations[all_observations['Sector'].isin(flaring_sectors)]
    non_flaring_observations = sector_one_to_three_observations[
        ~sector_one_to_three_observations[TessColumnName.TIC_ID].isin(
            flaring_metadata_data_frame[MetadataColumnName.TIC_ID])]
    non_flaring_tic_ids = non_flaring_observations[TessColumnName.TIC_ID].unique()
    non_flaring_data_frame = pd.DataFrame({
        MetadataColumnName.TIC_ID: non_flaring_tic_ids,
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE: pd.NA,
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT: pd.NA,
    })
    return non_flaring_data_frame


def download_flare_metadata_csv(non_flaring_limit: int = 10000) -> None:
    """
    Downloads the metadata for the flare application to a CSV.
    """
    flaring_target_metadata_data_frame = download_maximilian_gunther_metadata_data_frame()
    non_flaring_target_metadata_data_frame = \
        download_data_frame_of_tess_non_flaring_metadata_from_sectors_with_flaring_metadata(
            flaring_target_metadata_data_frame)
    tess_data_interface = TessDataInterface()
    flaring_sectors = []
    for index, row in flaring_target_metadata_data_frame.iterrows():
        target_sectors = tess_data_interface.get_sectors_target_appears_in(int(row[MetadataColumnName.TIC_ID]))
        flaring_sectors.extend(target_sectors)
    flaring_sectors = [sector for sector in set(flaring_sectors) if sector <= 26]
    labeled_tuple_list: List[Tuple[int, int, float, float, float, int]] = []
    flaring_tic_id_and_sector_list: List[Tuple[int, int]] = []
    for tic_id in flaring_target_metadata_data_frame[MetadataColumnName.TIC_ID].values:
        sectors = tess_data_interface.get_sectors_target_appears_in(tic_id)
        for sector in sectors:
            if sector in flaring_sectors:
                tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)  # TODO: Don't repeat this call below.
                luminosity__solar_units = tic_row['lum']
                if np.isnan(luminosity__solar_units) or luminosity__solar_units == 0:
                    continue
                flaring_tic_id_and_sector_list.append((tic_id, sector))
    random = Random()
    random.seed(0)
    random.shuffle(flaring_tic_id_and_sector_list)
    flaring_splits = split_tic_id_and_sector_list_equally(flaring_tic_id_and_sector_list, number_of_splits=10)
    for split_index, tic_id_and_sector_split in enumerate(flaring_splits):
        for tic_id, sector in tic_id_and_sector_split:
            row = flaring_target_metadata_data_frame[
                flaring_target_metadata_data_frame[MetadataColumnName.TIC_ID] == tic_id].iloc[0]
            tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)
            luminosity__solar_units = tic_row['lum']
            if np.isnan(luminosity__solar_units) or luminosity__solar_units == 0:
                continue
            luminosity__log_10_solar_units = np.log10(luminosity__solar_units)
            labeled_tuple_list.append(
                (tic_id, sector,
                 row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                 row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT],
                 luminosity__log_10_solar_units,
                 split_index)
            )
    non_flaring_tic_id_and_sector_list: List[Tuple[int, int]] = []
    non_flaring_tic_ids = non_flaring_target_metadata_data_frame[MetadataColumnName.TIC_ID].values
    random.shuffle(non_flaring_tic_ids)
    # 5x should be enough that without luminosity it should still reach the limit.
    for tic_id in non_flaring_tic_ids[:non_flaring_limit*5]:
        sectors = tess_data_interface.get_sectors_target_appears_in(tic_id)
        for sector in sectors:
            if sector in flaring_sectors:
                non_flaring_tic_id_and_sector_list.append((tic_id, sector))
    random.shuffle(non_flaring_tic_id_and_sector_list)
    non_flaring_tic_id_and_sector_list_with_luminosity: List[Tuple[int, int]] = []
    for tic_id, sector in non_flaring_tic_id_and_sector_list:
        tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)  # TODO: Don't repeat this call below.
        luminosity__solar_units = tic_row['lum']
        if np.isnan(luminosity__solar_units) or luminosity__solar_units == 0:
            continue
        non_flaring_tic_id_and_sector_list_with_luminosity.append((tic_id, sector))
        if len(non_flaring_tic_id_and_sector_list_with_luminosity) >= non_flaring_limit:
            break
    non_flaring_splits = split_tic_id_and_sector_list_equally(non_flaring_tic_id_and_sector_list_with_luminosity,
                                                              number_of_splits=10)
    for split_index, tic_id_and_sector_split in enumerate(non_flaring_splits):
        for tic_id, sector in tic_id_and_sector_split:
            tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)
            luminosity__solar_units = tic_row['lum']
            if np.isnan(luminosity__solar_units) or luminosity__solar_units == 0:
                continue
            luminosity__log_10_solar_units = np.log10(luminosity__solar_units)
            labeled_tuple_list.append((tic_id, sector, np.NaN, np.NaN, luminosity__log_10_solar_units, split_index))
    metadata_data_frame = pd.DataFrame(labeled_tuple_list,
                                       columns=[MetadataColumnName.TIC_ID,
                                                MetadataColumnName.SECTOR,
                                                MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE,
                                                MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT,
                                                MetadataColumnName.LUMINOSITY__LOG_10_SOLAR_UNITS,
                                                MetadataColumnName.SPLIT])
    metadata_data_frame.to_csv(metadata_csv_path, index=False, na_rep='NA')


if __name__ == '__main__':
    download_flare_metadata_csv()
