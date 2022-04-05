import json
from typing import Optional

import lightkurve
from astropy import units
from astropy.coordinates import SkyCoord, Angle
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia
from retrying import retry

from ramjet.data_interface.tess_data_interface import is_common_mast_connection_error
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve, tess_pixel_angular_size, \
    CentroidAlgorithmFailedError


class GaiaAwareTessFfiLightCurve(TessFfiLightCurve):
    Gaia.ROW_LIMIT = -1
    query_string = """
    SELECT *
    FROM gaiadr2.vari_classifier_result
    INNER JOIN gaiadr2.gaia_source USING (source_id)
    """
    gaia_variable_target_job = Gaia.launch_job_async(query=query_string)
    gaia_variable_target_result = gaia_variable_target_job.get_results()
    gaia_variable_target_data_frame: pd.DataFrame = gaia_variable_target_result.to_pandas()
    gaia_rr_lyrae_target_data_frame = gaia_variable_target_data_frame[
        gaia_variable_target_data_frame['best_class_name'].isin(['ARRD', 'RRC', 'RRAB', 'RRD'])]

    def __init__(self):
        super().__init__()

    @retry(retry_on_exception=is_common_mast_connection_error)
    def separation_to_nearest_gaia_rr_lyrae_within_separation(self, sky_coord: SkyCoord,
                                                              maximum_separation: Angle(21, unit=units.arcsecond)
                                                              ) -> Optional[Angle]:
        gaia_job = Gaia.cone_search_async(sky_coord, radius=maximum_separation)
        gaia_result = gaia_job.get_results()
        gaia_region_data_frame = gaia_result.to_pandas()
        rr_lyrae_gaia_region_data_frame = gaia_region_data_frame[
            gaia_region_data_frame['source_id'].isin(self.gaia_rr_lyrae_target_data_frame['source_id'])]
        if rr_lyrae_gaia_region_data_frame.shape[0] == 0:
            return None
        try:
            closet_rr_lyrae_row = rr_lyrae_gaia_region_data_frame.iloc[0]
            closet_rr_lyrae_coordinates = SkyCoord(ra=closet_rr_lyrae_row['ra'], dec=closet_rr_lyrae_row['dec'],
                                                   unit=units.deg)
            return sky_coord.separation(closet_rr_lyrae_coordinates)
        except IndexError:
            return None


def filter_rr_lyrae(results_data_frame: pd.DataFrame) -> pd.DataFrame:
    dropped_by_known_count = 0
    dropped_by_centroid_offset_count = 0
    print('Loading light curves...', flush=True)

    def light_curve_from_row(row: pd.Series) -> TessFfiLightCurve:
        light_curve_path = Path(row['light_curve_path'])
        # Hack to fix changes to Adapt.
        old_adapt_path = Path('/att/gpfsfs/home/golmsche')
        new_adapt_path = Path('/adapt/nobackup/people/golmsche')
        if old_adapt_path in light_curve_path.parents:
            sub_path = light_curve_path.relative_to(old_adapt_path)
            light_curve_path = new_adapt_path.joinpath(sub_path)
        return TessFfiLightCurve.from_path(light_curve_path)

    gaia_rr_lyrae_minimum_period = 0.2009529790117998
    gaia_rr_lyrae_maximum_period = 0.9975636975622972
    results_data_frame['light_curve'] = results_data_frame.apply(light_curve_from_row, axis=1)
    print('Calculating variability...', flush=True)
    for index, row in results_data_frame.iterrows():
        print(index, end='\r', flush=True)
        light_curve = row['light_curve']
        try:
            nearest_known_separation = light_curve.separation_to_nearest_gaia_rr_lyrae_within_separation(
                light_curve.sky_coord, tess_pixel_angular_size * 2)
            if nearest_known_separation is not None:
                results_data_frame.drop(index, inplace=True)
                dropped_by_known_count += 1
                continue
        except json.decoder.JSONDecodeError:
            results_data_frame.drop(index, inplace=True)
            dropped_by_centroid_offset_count += 1
            continue
        try:
            separation_to_variability_photometric_centroid = \
                light_curve.get_angular_distance_to_variability_photometric_centroid(
                    minimum_period=gaia_rr_lyrae_minimum_period, maximum_period=gaia_rr_lyrae_maximum_period)
        except (CentroidAlgorithmFailedError, lightkurve.search.SearchError, ValueError):
            results_data_frame.drop(index, inplace=True)
            dropped_by_centroid_offset_count += 1
            continue
        if separation_to_variability_photometric_centroid > tess_pixel_angular_size:
            results_data_frame.drop(index, inplace=True)
            dropped_by_centroid_offset_count += 1
            continue

    print('Adding additional columns...', flush=True)

    def sky_coord_from_row(row: pd.Series) -> SkyCoord:
        return row['light_curve'].sky_coord

    def magnitude_from_row(row: pd.Series) -> float:
        return row['light_curve'].tess_magnitude

    def tic_id_from_row(row: pd.Series) -> float:
        return row['light_curve'].tic_id

    def sector_from_row(row: pd.Series) -> float:
        return row['light_curve'].sector

    results_data_frame['tic_id'] = results_data_frame.apply(tic_id_from_row, axis=1)
    results_data_frame['sector'] = results_data_frame.apply(sector_from_row, axis=1)
    tic_id_duplicated_count = results_data_frame.shape[0]
    results_data_frame = results_data_frame.drop_duplicates(['tic_id'])
    tic_id_deduplicated_count = results_data_frame.shape[0]
    results_data_frame['sky_coord'] = results_data_frame.apply(sky_coord_from_row, axis=1)
    results_data_frame['magnitude'] = results_data_frame.apply(magnitude_from_row, axis=1)
    duplicated_count = results_data_frame.shape[0]
    dropped_due_to_brighter_target_nearby = 0
    print('Comparing separations...', flush=True)
    for index, row in results_data_frame.iterrows():
        print(index, end='\r', flush=True)
        data_frame_excluding_row = results_data_frame.drop(index)

        def separation_to_current(other_row: pd.Series) -> Angle:
            return row['sky_coord'].separation(other_row['sky_coord'])

        data_frame_excluding_row['separation'] = data_frame_excluding_row.apply(separation_to_current, axis=1)

        def separation_less_than_tess_pixel(row: pd.Series) -> bool:
            return row['separation'] < tess_pixel_angular_size

        competing_data_frame = data_frame_excluding_row[
            data_frame_excluding_row.apply(separation_less_than_tess_pixel, axis=1)]
        if competing_data_frame.shape[0] == 0:
            continue
        if row['magnitude'] is None:
            results_data_frame.drop(index, inplace=True)
            dropped_due_to_brighter_target_nearby += 1
            continue
        if (competing_data_frame[competing_data_frame['magnitude'] <= row['magnitude']]).shape[0] > 0:
            results_data_frame.drop(index, inplace=True)
            dropped_due_to_brighter_target_nearby += 1
            continue
    deduplicated_count = results_data_frame.shape[0]
    print(f'Dropped as known: {dropped_by_known_count}')
    print(f'Dropped as centroid offset: {dropped_by_centroid_offset_count}')
    print(f'Dropped as TIC ID duplicates: {tic_id_duplicated_count - tic_id_deduplicated_count}')
    print(f'Dropped as position duplicates: {duplicated_count - deduplicated_count}')
    print(f'Dropped due to brighter target nearby: {dropped_due_to_brighter_target_nearby}')

    def ra_from_row(row: pd.Series) -> float:
        return row['sky_coord'].ra

    results_data_frame['ra'] = results_data_frame.apply(ra_from_row, axis=1)

    def dec_from_row(row: pd.Series) -> float:
        return row['sky_coord'].dec

    results_data_frame['dec'] = results_data_frame.apply(dec_from_row, axis=1)

    def period_from_row(row: pd.Series) -> float:
        light_curve_ = row['light_curve']
        fold_period = light_curve_.variability_period
        return fold_period

    def period_epoch_from_row(row: pd.Series) -> float:
        light_curve_ = row['light_curve']
        fold_epoch = light_curve_.variability_period_epoch
        return fold_epoch

    results_data_frame['period'] = results_data_frame.apply(period_from_row, axis=1)
    results_data_frame['period_epoch'] = results_data_frame.apply(period_epoch_from_row, axis=1)
    results_data_frame.drop('sky_coord', axis=1, inplace=True)
    results_data_frame.drop('index', axis=1, inplace=True)
    results_data_frame.drop('light_curve', axis=1, inplace=True)
    return results_data_frame


if __name__ == '__main__':
    infer_results_path = Path('/att/gpfsfs/briskfs01/ppl/golmsche/generalized-photometric-neural-network-experiments/'
                              'logs/FfiHades_corrected_non_rrl_label_no_bn_2022_02_03_23_52_12/'
                              'infer_results_2022-02-06-13-21-41.csv')
    filtered_results_path = infer_results_path.parent.joinpath(f'filtered_{infer_results_path.name}')
    results_data_frame = pd.read_csv(infer_results_path)
    results_data_frame = results_data_frame.head(10_000)
    results_data_frame = filter_rr_lyrae(results_data_frame)
    results_data_frame.to_csv(filtered_results_path, index=False)
