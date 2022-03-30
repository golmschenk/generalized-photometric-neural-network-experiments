from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord, ICRS
from astroquery.mast import Catalogs, Observations
from astroquery.vizier import Vizier
from astropy import units
from bokeh.io import show
from bokeh.palettes import Category10
from bokeh.plotting import Figure
from astroquery.gaia import Gaia
from retrying import retry

from ramjet.data_interface.tess_data_interface import is_common_mast_connection_error
from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve
from ramjet.photometric_database.tess_light_curve import TessLightCurve

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class GcvsColumnName(StrEnum):
    VARIABLE_TYPE_STRING = 'VarType'
    RA = 'RAJ2000'
    DEC = 'DEJ2000'


def has_gcvs_type(var_type_string: str, labels: List[str]) -> bool:
    var_type_string_without_uncertainty_flags = var_type_string.replace(':', '')
    variable_type_flags = var_type_string_without_uncertainty_flags.split('+')
    for variable_type_flag in variable_type_flags:
        if variable_type_flag in labels:
            return True
    return False


def get_gcvs_catalog_entries_for_labels(labels: List[str]) -> pd.DataFrame:
    gcvs_catalog_astropy_table = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1).query_constraints()[0]
    gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, labels)

    label_mask = gcvs_catalog_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    data_frame_of_classes = gcvs_catalog_data_frame[label_mask]
    return data_frame_of_classes


def get_gcvs_catalog_entries_without_labels(labels: List[str]) -> pd.DataFrame:
    gcvs_catalog_astropy_table = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1).query_constraints()[0]
    gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()

    def filter_function(var_type_string):
        return has_gcvs_type(var_type_string, labels)

    label_mask = gcvs_catalog_data_frame[GcvsColumnName.VARIABLE_TYPE_STRING].apply(filter_function)
    data_frame_of_classes = gcvs_catalog_data_frame[~label_mask]
    return data_frame_of_classes


@retry(retry_on_exception=is_common_mast_connection_error)
def get_tic_id_for_gcvs_row(gcvs_row: pd.Series) -> Optional[int]:
    # TODO: should we really search so much of the pixel size?
    half_tess_pixel_fov = Angle(10.5, unit=units.arcsecond)
    try:
        gcvs_coordinates = SkyCoord(ra=gcvs_row[GcvsColumnName.RA], dec=gcvs_row[GcvsColumnName.DEC],
                                    unit=(units.hourangle, units.deg), equinox='J2000')
    except ValueError:  # TODO: Catching that we didn't find one. Catching all ValueErrors might be a bit too general.
        return None
    region_results = Catalogs.query_region(gcvs_coordinates, radius=half_tess_pixel_fov, catalog='TIC',
                                           columns=['**']).to_pandas()
    # TODO: Perhaps filter on magnitudes max and min? Might not matter though, so long as it's close to the original
    # TODO: target, as the photometric data will not look different.
    if region_results.shape[0] != 0:
        # First index should be the closest.
        return int(region_results['ID'].iloc[0])
    else:
        return None

@retry(retry_on_exception=is_common_mast_connection_error)
def get_tic_id_for_gaia_row(gaia_row: pd.Series) -> Optional[int]:
    # TODO: should we really search so much of the pixel size?
    half_tess_pixel_fov = Angle(10.5, unit=units.arcsecond)
    try:
        gaia_coordinates = SkyCoord(ra=gaia_row['ra'], dec=gaia_row['dec'], unit=units.deg)
    except ValueError:  # TODO: Catching that we didn't find one. Catching all ValueErrors might be a bit too general.
        return None
    gaia_coordinates = gaia_coordinates.transform_to(ICRS)
    region_results = Catalogs.query_region(gaia_coordinates, radius=half_tess_pixel_fov, catalog='TIC',
                                           columns=['**']).to_pandas()
    # TODO: Perhaps filter on magnitudes max and min? Might not matter though, so long as it's close to the original
    # TODO: target, as the photometric data will not look different.
    if region_results.shape[0] != 0:
        # First index should be the closest.
        return int(region_results['ID'].iloc[0])
    else:
        return None


def download_gcvs_metadata():
    # gcvs_catalog_astropy_table = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1).query_constraints()[0]
    # gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()
    rr_lyrae_labels = ['RR', 'RR(B)', 'RRAB', 'RRC']
    rr_lyrae_data_frame = get_gcvs_catalog_entries_for_labels(rr_lyrae_labels)
    # cepheid_labels = ['CEP', 'CEP(B)', 'DCEP']
    # cepheid_data_frame = get_gcvs_catalog_entries_for_labels(cepheid_labels)
    tic_id = get_tic_id_for_gcvs_row(rr_lyrae_data_frame.iloc[0])

    period_histogram_figure = Figure()
    periods = rr_lyrae_data_frame['Period'].dropna().values
    histogram_values, bin_edges = np.histogram(periods, bins=50,
                                               density=True)
    fill_alpha = 0.8
    line_alpha = 0.9
    period_histogram_figure.quad(top=histogram_values, bottom=0,
                                left=bin_edges[:-1], right=bin_edges[1:],
                                fill_alpha=fill_alpha, color=Category10[10][0],
                                line_alpha=line_alpha)
    show(period_histogram_figure)
    print(np.mean(periods))
    print(np.median(periods))
    print(np.min(periods))
    print(np.max(periods))



def download_gaia_metadata_csv():
    Gaia.ROW_LIMIT = -1
    query_string = """
    SELECT *
    FROM gaiadr2.vari_classifier_result
    INNER JOIN gaiadr2.gaia_source USING (source_id)
    """
    gaia_variable_targets_csv_path = Path('data/variables/gaia_variable_targets.csv')
    gaia_variable_target_job = Gaia.launch_job_async(query=query_string)
    gaia_variable_target_result = gaia_variable_target_job.get_results()
    gaia_variable_target_data_frame: pd.DataFrame = gaia_variable_target_result.to_pandas()
    gaia_variable_target_data_frame.to_csv(gaia_variable_targets_csv_path, index=False)


def download_rr_lyrae_gaia_tess_metadata_csv():
    Gaia.ROW_LIMIT = -1
    query_string = """
        SELECT *
        FROM gaiadr2.vari_classifier_result
        INNER JOIN gaiadr2.gaia_source USING (source_id)
        """
    gaia_rr_lyrae_targets_csv_path = Path('data/variables/gaia_tess_rr_lyrae_targets.csv')
    gaia_variable_target_job = Gaia.launch_job_async(query=query_string)
    gaia_variable_target_result = gaia_variable_target_job.get_results()
    gaia_variable_target_data_frame: pd.DataFrame = gaia_variable_target_result.to_pandas()
    rr_lyrae_labels = ['ARRD', 'RRC', 'RRAB', 'RRD']
    gaia_rr_lyrae_target_data_frame = gaia_variable_target_data_frame[gaia_variable_target_data_frame['best_class_name'].isin(rr_lyrae_labels)]
    gaia_rr_lyrae_target_data_frame.to_csv('quick.csv')
    # gaia_rr_lyrae_target_data_frame = pd.read_csv('quick.csv')

    def light_curve_from_row(row: pd.Series) -> TessLightCurve:
        light_curve = TessLightCurve()
        light_curve.tic_id = get_tic_id_for_gaia_row(row)
        return light_curve

    gaia_rr_lyrae_target_data_frame['light_curve'] = gaia_rr_lyrae_target_data_frame.apply(light_curve_from_row, axis=1)

    def sky_coord_from_row(row: pd.Series) -> SkyCoord:
        return row['light_curve'].sky_coord

    def magnitude_from_row(row: pd.Series) -> float:
        return row['light_curve'].tess_magnitude

    def tic_id_from_row(row: pd.Series) -> float:
        return row['light_curve'].tic_id

    gaia_rr_lyrae_target_data_frame['tic_id'] = gaia_rr_lyrae_target_data_frame.apply(tic_id_from_row, axis=1)
    gaia_rr_lyrae_target_data_frame['sky_coord'] = gaia_rr_lyrae_target_data_frame.apply(sky_coord_from_row, axis=1)
    gaia_rr_lyrae_target_data_frame['magnitude'] = gaia_rr_lyrae_target_data_frame.apply(magnitude_from_row, axis=1)
    gaia_rr_lyrae_target_data_frame.to_csv(gaia_rr_lyrae_targets_csv_path, index=False)

def download_gaia_rr_lyrae_metadata_to_csv():
    Gaia.ROW_LIMIT = -1
    query_string = """
        SELECT *
        FROM gaiadr2.vari_rrlyrae
        """
    gaia_rr_lyrae_metadata_csv_path = Path('data/variables/gaia_rr_lyrae_metadata.csv')
    gaia_rr_lyrae_metadata_job = Gaia.launch_job_async(query=query_string)
    gaia_rr_lyrae_metadata_result = gaia_rr_lyrae_metadata_job.get_results()
    gaia_rr_lyrae_metadata_data_frame: pd.DataFrame = gaia_rr_lyrae_metadata_result.to_pandas()
    gaia_rr_lyrae_metadata_data_frame.to_csv(gaia_rr_lyrae_metadata_csv_path)


def download_tic_rows_for_gaia_rr_lyrae_source_ids():
    Observations.TIMEOUT = 2000
    Observations.PAGESIZE = 3000
    Catalogs.TIMEOUT = 2000
    Catalogs.PAGESIZE = 3000
    try:  # Temporary fix for astroquery's update of timeout and pagesize locations.
        Observations._portal_api_connection.TIMEOUT = 2000
        Observations._portal_api_connection.PAGESIZE = 3000
        Catalogs._portal_api_connection.TIMEOUT = 2000
        Catalogs._portal_api_connection.PAGESIZE = 3000
    except AttributeError:
        pass

    gaia_variable_target_data_frame = pd.read_csv('data/variables/gaia_rr_lyrae_metadata.csv')
    gaia_source_ids = gaia_variable_target_data_frame['source_id'].values
    tic_rows_for_gaia_source_ids_astropy_table = Catalogs.query_criteria(catalog="TIC", GAIA=gaia_source_ids)
    tic_rows_for_gaia_source_ids_data_frame = tic_rows_for_gaia_source_ids_astropy_table.to_pandas()
    tic_rows_for_gaia_source_ids_data_frame.to_csv('data/variables/tic_rows_for_gaia_rr_lyrae_source_ids.csv')


if __name__ == '__main__':
    download_gcvs_metadata()
    download_rr_lyrae_gaia_tess_metadata_csv()
