from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy import units
from bokeh.io import show
from bokeh.palettes import Category10
from bokeh.plotting import Figure

from ramjet.data_interface.tess_data_interface import TessDataInterface

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


def download_metadata():
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
    pass


if __name__ == '__main__':
    download_metadata()
