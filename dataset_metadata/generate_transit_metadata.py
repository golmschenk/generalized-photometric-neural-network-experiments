"""
A script to generate the metadata for the experiments.
"""
from collections import defaultdict

from typing import List, Tuple, Dict

from ramjet.data_interface.tess_toi_data_interface import TessToiDataInterface, ExofopDisposition, ToiColumns


def download_tess_primary_mission_confirmed_exofop_planet_transit_tic_id_and_sector_list() -> List[Tuple[int, int]]:
    """
    Downloads from ExoFOP a list of `(tic_id, sector)` tuples for any confirmed planets in the TESS primary mission
    data (sectors 1-26 inclusive).

    :return: The list of TIC ID and sector pairs.
    """
    tic_id_and_sector_list = []
    tess_toi_data_interface = TessToiDataInterface()
    toi_dispositions = tess_toi_data_interface.toi_dispositions
    confirmed_planet_disposition_labels = [ExofopDisposition.CONFIRMED_PLANET.value,
                                           ExofopDisposition.KEPLER_CONFIRMED_PLANET.value]
    confirmed_planet_dispositions = toi_dispositions[
        toi_dispositions[ToiColumns.disposition.value].isin(confirmed_planet_disposition_labels)]
    primary_mission_confirmed_planet_dispositions = confirmed_planet_dispositions[
        (confirmed_planet_dispositions[ToiColumns.sector.value] >= 1) &
        (confirmed_planet_dispositions[ToiColumns.sector.value] <= 26)]
    for index, row in primary_mission_confirmed_planet_dispositions.iterrows():
        tic_id_and_sector_list.append((row[ToiColumns.tic_id.value], row[ToiColumns.sector.value]))
    return tic_id_and_sector_list


def get_tic_id_sorted_count_dictionary(tic_id_and_sector_list: List[Tuple[int, int]]
                                       ) -> Dict[int, int]:
    """
    Produces a dictionary mapping each TIC ID to the count of light curves belonging to it, sorted with the largest
    counts first.

    :param tic_id_and_sector_list: The list of TIC ID and sector tuples of the available light curves.
    :return:
    """
    tic_id_count_dictionary = defaultdict(int)  # Make the default value for a new key be zero.
    for tic_id, sector in tic_id_and_sector_list:
        tic_id_count_dictionary[tic_id] += 1
    sorted_tic_id_count_dictionary = dict(sorted(tic_id_count_dictionary.items(), key=lambda item: item[1]))
    return sorted_tic_id_count_dictionary


def generate_transit_metadata() -> None:
    tic_id_and_sector_list = download_tess_primary_mission_confirmed_exofop_planet_transit_tic_id_and_sector_list()
    tic_id_sorted_count_dictionary = get_tic_id_sorted_count_dictionary(tic_id_and_sector_list)


if __name__ == '__main__':
    generate_transit_metadata()
