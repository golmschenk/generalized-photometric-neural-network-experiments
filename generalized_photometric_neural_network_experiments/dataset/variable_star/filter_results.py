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
        # TODO: unnecessary repeats of the TIC ID download.
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).sky_coord

    def magnitude_from_row(row: pd.Series) -> float:
        # TODO: unnecessary repeats of the TIC ID download.
        return TessFfiLightCurve.from_path(Path(row['light_curve_path'])).tess_magnitude

    tic_id_duplicated_count = results_data_frame.shape[0]
    results_data_frame = results_data_frame.drop_duplicates(['tic_id'])
    tic_id_deduplicated_count = results_data_frame.shape[0]
    results_data_frame['sky_coord'] = results_data_frame.apply(sky_coord_from_row, axis=1)
    results_data_frame['magnitude'] = results_data_frame.apply(magnitude_from_row, axis=1)
    duplicated_count = results_data_frame.shape[0]
    for index, row in results_data_frame.iterrows():
        print(index)
        data_frame_excluding_row = results_data_frame.drop(index)
        def separation_to_current(other_row: pd.Series) -> Angle:
            return row['sky_coord'].separation(other_row['sky_coord'])
        data_frame_excluding_row['separation'] = data_frame_excluding_row.apply(separation_to_current)
        competing_data_frame = data_frame_excluding_row[
            data_frame_excluding_row['separation'] < tess_pixel_angular_size]
        if competing_data_frame.shape[0] == 0:
            continue
        if row['magnitude'] is None:
            results_data_frame.drop(index, inplace=True)
            continue
        if (competing_data_frame[competing_data_frame['magnitude'] <= row['magnitude']]).shape[0] > 0:
            results_data_frame.drop(index, inplace=True)
            continue
    deduplicated_count = results_data_frame.shape[0]
    print(f'Dropped as known: {dropped_by_known_count}')
    print(f'Dropped as centroid offset: {dropped_by_centroid_offset_count}')
    print(f'Dropped as TIC ID duplicates: {tic_id_duplicated_count - tic_id_deduplicated_count}')
    print(f'Dropped as duplicates: {duplicated_count - deduplicated_count}')
    return results_data_frame
