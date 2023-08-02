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
from ramjet.photometric_database.tess_light_curve import CentroidAlgorithmFailedError, TessLightCurve

import lightkurve as lk

lk.conf.cache_dir = '/explore/nobackup/people/golmsche/.lightkurve'


class ShortPeriodTessFfiLightCurve(TessFfiLightCurve):
    # TODO: This is a bit too much of a hack that relies on overriding a random part of the regular light curve methods.
    def get_variability_phase_folding_parameters_and_folding_lightkurve_light_curves(
            self, minimum_period: Optional[float] = None, maximum_period: Optional[float] = None):
        median_time_step = np.median(np.diff(self.times[~np.isnan(self.times)]))
        lightkurve_light_curve = self.to_lightkurve()
        inlier_lightkurve_light_curve = lightkurve_light_curve.remove_outliers(sigma=3)
        periodogram = LombScarglePeriodogram.from_lightcurve(inlier_lightkurve_light_curve, oversample_factor=3,
                                                             minimum_period=minimum_period,
                                                             maximum_period=maximum_period)
        periods__days = periodogram.period.to(units.d).value
        powers = periodogram.power.value
        longest_period_index_near_max_power = np.argwhere(powers > 0.9 * periodogram.max_power)[0, -1]
        while (longest_period_index_near_max_power < len(powers) - 1 and
               powers[longest_period_index_near_max_power + 1] > powers[longest_period_index_near_max_power]):
            longest_period_index_near_max_power += 1
        longest_period_near_max_power = periods__days[longest_period_index_near_max_power]
        self._variability_power = powers[longest_period_index_near_max_power]
        folded_lightkurve_light_curve = inlier_lightkurve_light_curve.fold(period=longest_period_near_max_power,
                                                                           wrap_phase=longest_period_near_max_power)
        fold_period = folded_lightkurve_light_curve.period.value
        time_bin_size = fold_period / 25
        binned_folded_lightkurve_light_curve = folded_lightkurve_light_curve.bin(time_bin_size=time_bin_size,
                                                                                 aggregate_func=np.nanmedian)
        minimum_bin_index = np.nanargmin(binned_folded_lightkurve_light_curve.flux.value)
        maximum_bin_index = np.nanargmax(binned_folded_lightkurve_light_curve.flux.value)
        minimum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[minimum_bin_index]
        maximum_bin_phase = binned_folded_lightkurve_light_curve.phase.value[maximum_bin_index]
        fold_epoch = inlier_lightkurve_light_curve.time.value[0]
        return (longest_period_near_max_power, fold_epoch, time_bin_size, minimum_bin_phase, maximum_bin_phase,
                inlier_lightkurve_light_curve, periodogram, folded_lightkurve_light_curve)


class FilterProcesser:
    def __init__(self):
        self.dropped_by_known_count = 0
        self.dropped_by_centroid_offset_count = 0
        self.temporary_directory = Path('filter_temporary')
        self.tic_id_duplicated_count = None
        self.tic_id_deduplicated_count = None
        self.duplicated_count = None
        self.dropped_due_to_brighter_target_nearby = 0

    def filter_results(self, results_data_frame: pd.DataFrame) -> pd.DataFrame:
        temporary_file0 = self.temporary_directory.joinpath('temporary_file0.pkl')
        if not temporary_file0.exists():
            self.add_light_curves_to_data_frame(results_data_frame)
            results_data_frame.to_pickle(temporary_file0)
        else:
            print(f'Loading from {temporary_file0}')
            results_data_frame = pd.pandas.read_pickle(temporary_file0)

        temporary_file1 = self.temporary_directory.joinpath('temporary_file1.pkl')
        if not temporary_file1.exists():
            self.check_varability(results_data_frame)
            results_data_frame.to_pickle(temporary_file1)
        else:
            print(f'Loading from {temporary_file1}')
            results_data_frame = pd.pandas.read_pickle(temporary_file1)

        temporary_file2 = self.temporary_directory.joinpath('temporary_file2.pkl')
        if not temporary_file2.exists():
            results_data_frame = self.add_additional_columns(results_data_frame)
            results_data_frame.to_pickle(temporary_file2)
        else:
            print(f'Loading from {temporary_file2}')
            results_data_frame = pd.pandas.read_pickle(temporary_file2)

        temporary_file3 = self.temporary_directory.joinpath('temporary_file3.pkl')
        if not temporary_file3.exists():
            results_data_frame = self.remove_duplicates_by_separation(results_data_frame)
            results_data_frame.to_pickle(temporary_file3)
        else:
            print(f'Loading from {temporary_file3}')
            results_data_frame = pd.pandas.read_pickle(temporary_file3)

        temporary_file4 = self.temporary_directory.joinpath('temporary_file4.pkl')
        if not temporary_file4.exists():
            self.clean_up(results_data_frame)
            results_data_frame.to_pickle(temporary_file4)
        else:
            print(f'Loading from {temporary_file4}')
            results_data_frame = pd.pandas.read_pickle(temporary_file4)

        return results_data_frame

    def clean_up(self, results_data_frame):
        print(f'Dropped as known: {self.dropped_by_known_count}')
        print(f'Dropped as centroid offset: {self.dropped_by_centroid_offset_count}')
        print(f'Dropped as TIC ID duplicates: {self.tic_id_duplicated_count - self.tic_id_deduplicated_count}')
        print(f'Dropped as position duplicates: {self.duplicated_count - self.deduplicated_count}')
        print(f'Dropped due to brighter target nearby: {self.dropped_due_to_brighter_target_nearby}')

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

        def power_from_row(row: pd.Series) -> float:
            light_curve_ = row['light_curve']
            fold_period = light_curve_._variability_power
            return fold_period

        def period_epoch_from_row(row: pd.Series) -> float:
            light_curve_ = row['light_curve']
            fold_epoch = light_curve_.variability_period_epoch
            return fold_epoch

        results_data_frame['period'] = results_data_frame.apply(period_from_row, axis=1)
        results_data_frame['power'] = results_data_frame.apply(power_from_row, axis=1)
        results_data_frame['period_epoch'] = results_data_frame.apply(period_epoch_from_row, axis=1)
        results_data_frame.drop('sky_coord', axis=1, inplace=True)
        results_data_frame.drop('index', axis=1, inplace=True)
        results_data_frame.drop('light_curve', axis=1, inplace=True)

    def remove_duplicates_by_separation(self, results_data_frame):
        print('Comparing separations...', flush=True)
        new_data_frame = results_data_frame.iloc[:0, :].copy()
        temporary_file3a_old = self.temporary_directory.joinpath('temporary_file3a_old.pkl')
        temporary_file3a_new = self.temporary_directory.joinpath('temporary_file3a_new.pkl')
        if temporary_file3a_new.exists():
            print(f'Loading from {temporary_file3a_old}')
            results_data_frame = pd.pandas.read_pickle(temporary_file3a_old)
            print(f'Loading from {temporary_file3a_new}')
            new_data_frame = pd.pandas.read_pickle(temporary_file3a_new)
        for index, row in results_data_frame.iterrows():
            if index not in results_data_frame.index:
                continue
            if index % 1000 == 0:
                results_data_frame.to_pickle(temporary_file3a_old)
                new_data_frame.to_pickle(temporary_file3a_new)

            print(f'index: {index}, size: {results_data_frame.shape[0]}', end='\r', flush=True)
            data_frame_excluding_row = results_data_frame.drop(index)

            def separation_to_current(other_row: pd.Series) -> Angle:
                return row['sky_coord'].separation(other_row['sky_coord'])

            data_frame_excluding_row['separation'] = data_frame_excluding_row.apply(separation_to_current, axis=1)

            def separation_less_than_tess_pixel(row: pd.Series) -> bool:
                return row['separation'] < tess_pixel_angular_size

            competing_data_frame = data_frame_excluding_row[
                data_frame_excluding_row.apply(separation_less_than_tess_pixel, axis=1)]
            if competing_data_frame.shape[0] == 0:
                new_data_frame.loc[index] = results_data_frame.loc[index]
                results_data_frame = results_data_frame.drop(index)
                continue
            if row['magnitude'] is None:
                results_data_frame = results_data_frame.drop(index)
                self.dropped_due_to_brighter_target_nearby += 1
                continue
            if (competing_data_frame[competing_data_frame['magnitude'] <= row['magnitude']]).shape[0] > 0:
                results_data_frame = results_data_frame.drop(index)
                self.dropped_due_to_brighter_target_nearby += 1
                continue
            for competing_index, competing_row in competing_data_frame.iterrows():
                results_data_frame = results_data_frame.drop(competing_index)
            new_data_frame.loc[index] = results_data_frame.loc[index]
            results_data_frame = results_data_frame.drop(index)

        self.deduplicated_count = results_data_frame.shape[0]
        return new_data_frame

    def add_additional_columns(self, results_data_frame):
        print('Adding additional columns...', flush=True)

        def sky_coord_from_row(row: pd.Series) -> SkyCoord:
            return row['light_curve'].sky_coord

        def magnitude_from_row(row: pd.Series) -> float:
            return row['light_curve'].tess_magnitude

        def tic_id_from_row(row: pd.Series) -> float:
            return row['light_curve'].tic_id

        def sector_from_row(row: pd.Series) -> float:
            return row['light_curve'].sector

        TessLightCurve.load_tic_rows_from_mast_for_list(results_data_frame['light_curve'].values)
        results_data_frame['tic_id'] = results_data_frame.apply(tic_id_from_row, axis=1)
        results_data_frame['sector'] = results_data_frame.apply(sector_from_row, axis=1)
        self.tic_id_duplicated_count = results_data_frame.shape[0]
        results_data_frame = results_data_frame.drop_duplicates(['tic_id'])
        self.tic_id_deduplicated_count = results_data_frame.shape[0]
        results_data_frame['sky_coord'] = results_data_frame.apply(sky_coord_from_row, axis=1)
        results_data_frame['magnitude'] = results_data_frame.apply(magnitude_from_row, axis=1)
        self.duplicated_count = results_data_frame.shape[0]
        return results_data_frame

    def check_varability(self, results_data_frame):
        temporary_file0a = self.temporary_directory.joinpath('temporary_file0a.pkl')
        if not temporary_file0a.exists():
            results_data_frame.to_pickle(temporary_file0a)
        else:
            print(f'Loading from {temporary_file0a}')
            results_data_frame = pd.pandas.read_pickle(temporary_file0a)
        results_data_frame['variability_processed'] = False
        print('Calculating variability...', flush=True)
        for index, row in results_data_frame.iterrows():
            if row['light_curve']._variability_period is not None:
                continue
            if index % 300 == 0:
                results_data_frame.to_pickle(temporary_file0a)
            results_data_frame.loc[index, 'variability_processed'] = True
            print(
                f'Index: {index}, Dropped: {self.dropped_by_centroid_offset_count}, Size: {results_data_frame.shape[0]}',
                flush=True)
            light_curve = row['light_curve']
            try:
                minimum_time = np.nanmin(light_curve.times)
                maximum_time = np.nanmax(light_curve.times)
                time_differences = np.diff(light_curve.times)
                minimum_time_step = np.nanmin(time_differences)
                # period_upper_limit = maximum_time - minimum_time
                period_upper_limit = 10
                period_lower_limit = 0.0208333
                separation_to_variability_photometric_centroid = \
                    light_curve.estimate_angular_distance_to_variability_photometric_centroid_from_ffi(
                        minimum_period=period_lower_limit, maximum_period=period_upper_limit)
            except (CentroidAlgorithmFailedError, lightkurve.search.SearchError) as error:
                results_data_frame.drop(index, inplace=True)
                self.dropped_by_centroid_offset_count += 1
                continue
            if separation_to_variability_photometric_centroid > tess_pixel_angular_size:
                results_data_frame.drop(index, inplace=True)
                self.dropped_by_centroid_offset_count += 1
                continue

    def add_light_curves_to_data_frame(self, results_data_frame):
        print('Loading light curves...', flush=True)

        def light_curve_from_row(row: pd.Series) -> TessFfiLightCurve:
            light_curve_path = Path(row['light_curve_path'])
            # Hack to fix changes to Adapt.
            old_adapt_path = Path('/att/gpfsfs/home/golmsche')
            new_adapt_path = Path('/explore/nobackup/people/golmsche')
            if old_adapt_path in light_curve_path.parents:
                sub_path = light_curve_path.relative_to(old_adapt_path)
                light_curve_path = new_adapt_path.joinpath(sub_path)
            return ShortPeriodTessFfiLightCurve.from_path(light_curve_path)

        results_data_frame['light_curve'] = results_data_frame.apply(light_curve_from_row, axis=1)


def main():
    infer_results_path = Path('/explore/nobackup/people/golmsche/generalized-photometric-neural-network-experiments/'
                              'logs/FfiHades_mixed_sine_sawtooth_2022_10_07_17_03_23/'
                              'infer_results_2022-12-11-19-35-34.csv')
    filtered_results_path = infer_results_path.parent.joinpath(f'filtered_{infer_results_path.name}')
    results_data_frame = pd.read_csv(infer_results_path)
    results_data_frame = results_data_frame.head(50_000)
    # results_data_frame = results_data_frame.head(30)
    filter_processor = FilterProcesser()
    results_data_frame = filter_processor.filter_results(results_data_frame)
    results_data_frame.to_csv(filtered_results_path, index=False)


if __name__ == '__main__':
    main()
