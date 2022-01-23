from astropy.coordinates import SkyCoord, Angle
from pathlib import Path

import pandas as pd

from ramjet.photometric_database.tess_ffi_light_curve import separation_to_nearest_gcvs_rr_lyrae_within_separation, \
    TessFfiLightCurve, tess_pixel_angular_size, CentroidAlgorithmFailedError


def filter_rr_lyrae(results_data_frame: pd.DataFrame) -> pd.DataFrame:
    dropped_by_known_count = 0
    dropped_by_centroid_offset_count = 0

    for index, row in results_data_frame.iterrows():
        print(index)
        light_curve_path = Path(row['light_curve_path'])
        light_curve = TessFfiLightCurve.from_path(Path(light_curve_path))
        nearest_known_separation = separation_to_nearest_gcvs_rr_lyrae_within_separation(
            light_curve.sky_coord, tess_pixel_angular_size * 3)
        if nearest_known_separation is not None:
            results_data_frame.drop(index, inplace=True)
            dropped_by_known_count += 1
            continue
        try:
            separation_to_variability_photometric_centroid = \
                light_curve.get_angular_distance_to_variability_photometric_centroid()
        except CentroidAlgorithmFailedError:
            results_data_frame.drop(index, inplace=True)
            dropped_by_centroid_offset_count += 1
            continue
        if separation_to_variability_photometric_centroid > tess_pixel_angular_size:
            results_data_frame.drop(index, inplace=True)
            dropped_by_centroid_offset_count += 1
            continue

    def sky_coord_from_row(row: pd.Series) -> SkyCoord:
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).sky_coord

    def magnitude_from_row(row: pd.Series) -> float:
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).tess_magnitude

    def tic_id_from_row(row: pd.Series) -> float:
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).tic_id

    def sector_from_row(row: pd.Series) -> float:
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).sector

    results_data_frame['tic_id'] = results_data_frame.apply(tic_id_from_row, axis=1)
    results_data_frame['sector'] = results_data_frame.apply(sector_from_row, axis=1)
    tic_id_duplicated_count = results_data_frame.shape[0]
    results_data_frame = results_data_frame.drop_duplicates(['tic_id'])
    tic_id_deduplicated_count = results_data_frame.shape[0]
    results_data_frame['sky_coord'] = results_data_frame.apply(sky_coord_from_row, axis=1)
    results_data_frame['magnitude'] = results_data_frame.apply(magnitude_from_row, axis=1)
    duplicated_count = results_data_frame.shape[0]
    dropped_due_to_brighter_target_nearby = 0
    for index, row in results_data_frame.iterrows():
        print(index)
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
    print(f'Dropped as duplicates: {duplicated_count - deduplicated_count}')
    print(f'Dropped due to brighter target nearby: {dropped_due_to_brighter_target_nearby}')
    def ra_from_row(row: pd.Series) -> float:
        return row['sky_coord'].ra
    results_data_frame['ra'] = results_data_frame.apply(ra_from_row, axis=1)
    def dec_from_row(row: pd.Series) -> float:
        return row['sky_coord'].dec
    results_data_frame['dec'] = results_data_frame.apply(dec_from_row, axis=1)
    results_data_frame.drop('sky_coord', axis=1, inplace=True)
    results_data_frame.drop('index', axis=1, inplace=True)
    def period_from_row(row: pd.Series) -> float:
        light_curve_ = TessFfiLightCurve.from_path(Path(row['light_curve_path']))
        fold_period = light_curve_.variability_period
        return fold_period
    def period_epoch_from_row(row: pd.Series) -> float:
        light_curve_ = TessFfiLightCurve.from_path(Path(row['light_curve_path']))
        fold_epoch = light_curve_.variability_period_epoch
        return fold_epoch
    results_data_frame['period'] = results_data_frame.apply(period_from_row, axis=1)
    results_data_frame['period_epoch'] = results_data_frame.apply(period_epoch_from_row, axis=1)
    return results_data_frame
