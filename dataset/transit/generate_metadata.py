"""
A script to generate the metadata for the experiments.
"""
import numpy as np
import pandas as pd
from random import Random
from collections import defaultdict
from typing import List, Tuple, Dict

from dataset.transit.names_and_paths import MetadataColumnName, TransitLabel, metadata_csv_path
from ramjet.data_interface.tess_data_interface import TessDataInterface, ColumnName as TessColumnName
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
    tess_data_interface = TessDataInterface()
    for index, row in primary_mission_confirmed_planet_dispositions.iterrows():
        tic_id = row[ToiColumns.tic_id.value]
        sector = row[ToiColumns.sector.value]
        # Check that the target appears in the 2 minute cadence data in that sector.
        if sector in tess_data_interface.get_sectors_target_appears_in(tic_id):
            tic_id_and_sector_list.append((tic_id, sector))
    return tic_id_and_sector_list


def get_tic_id_sorted_count_dictionary(tic_id_and_sector_list: List[Tuple[int, int]]
                                       ) -> Dict[int, int]:
    """
    Produces a dictionary mapping each TIC ID to the count of light curves belonging to it, sorted with the largest
    counts first.

    :param tic_id_and_sector_list: The list of TIC ID and sector tuples of the available light curves.
    :return: The sorted dictionary of TIC ID to counts.
    """
    tic_id_count_dictionary = defaultdict(int)  # Make the default value for a new key be zero.
    for tic_id, sector in tic_id_and_sector_list:
        tic_id_count_dictionary[tic_id] += 1
    sorted_tic_id_count_dictionary = dict(sorted(tic_id_count_dictionary.items(), key=lambda item: item[1],
                                                 reverse=True))
    return sorted_tic_id_count_dictionary


def split_tic_id_and_sector_list_equally(tic_id_and_sector_list: List[Tuple[int, int]], number_of_splits: int
                                         ) -> List[List[Tuple[int, int]]]:
    """
    Splits a list of TIC ID and sectors into equally sized splits while keeping different sectors of the same TIC ID in
    the same split.

    :param tic_id_and_sector_list: The list of TIC ID and sector tuples.
    :param number_of_splits: The number of equally sized splits to produce.
    :return: The list of splits, each of which is a list of TIC ID and sector tuples.
    """
    tic_id_sorted_count_dictionary = get_tic_id_sorted_count_dictionary(tic_id_and_sector_list)
    split_counts = np.zeros(shape=number_of_splits, dtype=np.int32)
    splits: List[List[Tuple[int, int]]] = [[] for _ in range(number_of_splits)]
    for tic_id_to_find in tic_id_sorted_count_dictionary.keys():
        smallest_split_index = np.argmin(split_counts)
        next_round_tic_id_and_sector_list = []
        for (tic_id, sector) in tic_id_and_sector_list:
            if tic_id == tic_id_to_find:
                splits[smallest_split_index].append((tic_id, sector))
                split_counts[smallest_split_index] += 1
            else:
                next_round_tic_id_and_sector_list.append((tic_id, sector))
        tic_id_and_sector_list = next_round_tic_id_and_sector_list
    return splits


def splits_to_labeled_tuple_list(tic_id_and_sector_splits: List[List[Tuple[int, int]]], label: str
                                 ) -> List[Tuple[int, int, str, int]]:
    """
    Convert the splits to a list of tuples with the split index and label being part of the tuple.

    :param tic_id_and_sector_splits: The TIC ID and sector split lists.
    :param label: The label to apply to all elements of all splits.
    :return: A list of tuples of the form `(tic_id, sector, label, split)`
    """
    labeled_tuple_list: List[Tuple[int, int, str, int]] = []
    for split_index, tic_id_and_sector_split in enumerate(tic_id_and_sector_splits):
        for tic_id, sector in tic_id_and_sector_split:
            labeled_tuple_list.append((tic_id, sector, label, split_index))
    return labeled_tuple_list


def add_splits_and_labels(tic_id_and_sector_list: List[Tuple[int, int]], number_of_splits: int, label: str
                          ) -> List[Tuple[int, int, str, int]]:
    """
    Adds split information, with equally sized splits keeping duplicate TIC IDs in the same split, and adds labels.

    :param tic_id_and_sector_list:
    :param number_of_splits: The number of equally sized splits to produce.
    :param label: The label to apply to all elements.
    :return: The list of tuples with each element in the form `(tic_id, sector, label, split)`.
    """
    tic_id_and_sector_splits = split_tic_id_and_sector_list_equally(tic_id_and_sector_list,
                                                                    number_of_splits=number_of_splits)
    tuple_list = splits_to_labeled_tuple_list(tic_id_and_sector_splits, label=label)
    return tuple_list


def download_tess_primary_mission_non_confirmed_nor_candidate_exofop_planet_list(limit: int = 10000
                                                                                 ) -> List[Tuple[int, int]]:
    """
    Downloads from MAST a list of `(tic_id, sector)` tuples for any targets which are not candidate nor confirmed
    ExoFOP planets in the TESS primary mission data (sectors 1-26 inclusive).

    :param limit: The limit for the size of the tuple list.
    :return: The list of TIC ID and sector pairs.
    """
    observations = TessDataInterface().get_all_two_minute_single_sector_observations()
    primary_mission_observations = observations[observations['Sector'] <= 26]
    tess_toi_data_interface = TessToiDataInterface()
    toi_dispositions = tess_toi_data_interface.toi_dispositions
    candidate_or_confirmed_planet_disposition_labels = [ExofopDisposition.CONFIRMED_PLANET.value,
                                                        ExofopDisposition.KEPLER_CONFIRMED_PLANET.value,
                                                        ExofopDisposition.PLANET_CANDIDATE.value]
    candidate_or_confirmed_planet_dispositions = toi_dispositions[
        toi_dispositions[ToiColumns.disposition.value].isin(candidate_or_confirmed_planet_disposition_labels)]
    candidate_or_confirmed_planet_tic_ids = candidate_or_confirmed_planet_dispositions[ToiColumns.tic_id.value].values
    non_candidate_nor_confirmed_observations = primary_mission_observations[
        ~observations[TessColumnName.TIC_ID].isin(candidate_or_confirmed_planet_tic_ids)]
    tic_id_and_sector_list = list(map(tuple, non_candidate_nor_confirmed_observations[
        [TessColumnName.TIC_ID, TessColumnName.SECTOR]].values))
    random = Random()
    random.seed(0)
    random.shuffle(tic_id_and_sector_list)
    reduced_tic_id_and_sector_list = tic_id_and_sector_list[:limit]
    return reduced_tic_id_and_sector_list


def generate_transit_metadata() -> None:
    """
    Generates the transit data set metadata.
    """
    planet_tic_id_and_sector_list = \
        download_tess_primary_mission_confirmed_exofop_planet_transit_tic_id_and_sector_list()
    planet_meta_data_list = add_splits_and_labels(planet_tic_id_and_sector_list, number_of_splits=10,
                                                  label=TransitLabel.PLANET)
    non_planet_tic_id_and_sector_list = download_tess_primary_mission_non_confirmed_nor_candidate_exofop_planet_list()
    non_planet_meta_data_list = add_splits_and_labels(non_planet_tic_id_and_sector_list, number_of_splits=10,
                                                      label=TransitLabel.NON_PLANET)
    metadata_data_frame = pd.DataFrame(planet_meta_data_list + non_planet_meta_data_list,
                                       columns=[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR,
                                                MetadataColumnName.LABEL, MetadataColumnName.SPLIT])
    metadata_data_frame.to_csv(metadata_csv_path, index=False)


if __name__ == '__main__':
    generate_transit_metadata()
