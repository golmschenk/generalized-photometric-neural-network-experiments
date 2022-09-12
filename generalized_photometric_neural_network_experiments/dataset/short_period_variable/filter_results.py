import json
from typing import Optional

import lightkurve
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord, Angle
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia
from lightkurve.periodogram import LombScarglePeriodogram
from retrying import retry

from generalized_photometric_neural_network_experiments.dataset.variable_star.download_metadata import \
    gaia_variable_targets_csv_path, download_gaia_variable_targets_metadata_csv, gaia_dr3_rr_lyrae_classes
from ramjet.data_interface.tess_data_interface import is_common_mast_connection_error
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve, tess_pixel_angular_size
from ramjet.photometric_database.tess_light_curve import CentroidAlgorithmFailedError


class ShortPeriodTessFfiLightCurve(TessFfiLightCurve):
    # TODO: This is a bit too much of a hack that relies on overriding a random part of the regular light curve methods.
    def get_variability_phase_folding_parameters_and_folding_lightkurve_light_curves(
            self, minimum_period: Optional[float] = None, maximum_period: Optional[float] = None):
        median_time_step = np.median(np.diff(self.times[~np.isnan(self.times)]))
        time_bin_size = median_time_step
        lightkurve_light_curve = self.to_lightkurve()
        inlier_lightkurve_light_curve = lightkurve_light_curve.remove_outliers(sigma=3)
        periodogram = LombScarglePeriodogram.from_lightcurve(inlier_lightkurve_light_curve, oversample_factor=3,
                                                             minimum_period=minimum_period,
                                                             maximum_period=maximum_period)
        periods__days = periodogram.period.to(units.d).value
        powers = periodogram.power.value
        longest_period_index_near_max_power = np.argwhere(powers > 0.9 * periodogram.max_power)[0, -1]
        while powers[longest_period_index_near_max_power + 1] > powers[longest_period_index_near_max_power]:
            longest_period_index_near_max_power += 1
        longest_period_near_max_power = periods__days[longest_period_index_near_max_power]
        folded_lightkurve_light_curve = inlier_lightkurve_light_curve.fold(period=longest_period_near_max_power,
                                                                           wrap_phase=longest_period_near_max_power)
        binned_folded_lightkurve_light_curve = folded_lightkurve_light_curve.bin(time_bin_size=time_bin_size,
                                                                                 aggregate_func=np.nanmedian)
        minimum_bin_index = np.nanargmin(binned_folded_lightkurve_light_curve.flux.value)
        maximum_bin_index = np.nanargmax(binned_folded_lightkurve_light_curve.flux.value)
        minimum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[minimum_bin_index]
        maximum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[maximum_bin_index]
        fold_epoch = inlier_lightkurve_light_curve.time.value[0]
        return (longest_period_near_max_power, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase,
                inlier_lightkurve_light_curve, periodogram, folded_lightkurve_light_curve)


def filter_results(results_data_frame: pd.DataFrame) -> pd.DataFrame:
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
        return ShortPeriodTessFfiLightCurve.from_path(light_curve_path)

    results_data_frame['light_curve'] = results_data_frame.apply(light_curve_from_row, axis=1)
    print('Calculating variability...', flush=True)
    for index, row in results_data_frame.iterrows():
        print(index, end='\r', flush=True)
        light_curve = row['light_curve']
        try:
            minimum_time = np.nanmin(light_curve.times)
            maximum_time = np.nanmax(light_curve.times)
            time_differences = np.diff(light_curve.times)
            minimum_time_step = np.nanmin(time_differences)
            period_upper_limit = maximum_time - minimum_time
            period_lower_limit = minimum_time_step / 2.1
            separation_to_variability_photometric_centroid = \
                light_curve.estimate_angular_distance_to_variability_photometric_centroid_from_ffi(
                    minimum_period=period_lower_limit, maximum_period=period_upper_limit)
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
    infer_results_path = Path('/adapt/nobackup/people/golmsche/generalized-photometric-neural-network-experiments/logs/'
                              'FfiHades_2022_09_06_21_23_14/infer_results_2022-09-07-15-21-50.csv')
    filtered_results_path = infer_results_path.parent.joinpath(f'filtered_{infer_results_path.name}')
    results_data_frame = pd.read_csv(infer_results_path)
    results_data_frame = results_data_frame.head(1_000)
    results_data_frame = filter_results(results_data_frame)
    results_data_frame.to_csv(filtered_results_path, index=False)
