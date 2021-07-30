"""
Visualization of the flare dataset.
"""

import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource, PrintfTickFormatter, Row
from bokeh.plotting import Figure
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from dataset.flare.names_and_paths import MetadataColumnName


def show_flare_frequency_distribution_plots() -> None:
    """
    Show some plots about the flare frequency distribution statistics.
    """
    metadata_data_frame = pd.read_csv('dataset_metadata/metadata.csv')
    flaring_metadata_data_frame = metadata_data_frame.dropna()
    flaring_metadata_data_frame['y_intercept'] = flaring_metadata_data_frame[
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
    flaring_metadata_data_frame['x_intercept'] = (
            -flaring_metadata_data_frame['y_intercept'] /
            flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE]
    )
    data_source = ColumnDataSource(flaring_metadata_data_frame)
    ffd_figure = Figure(x_range=(0, flaring_metadata_data_frame['x_intercept'].max()),
                        y_range=(0, flaring_metadata_data_frame['y_intercept'].max()))
    ffd_figure.segment(x0=0, y0='y_intercept', x1='x_intercept', y1=0, source=data_source, color='firebrick',
                       alpha=0.2)
    ffd_figure.yaxis.formatter = PrintfTickFormatter(format='10^%s')
    ffd_figure.xaxis.formatter = PrintfTickFormatter(format='10^%s')

    confidence = 0.68
    half_confidence = confidence / 2
    lower_bound_probability = 0.5 - half_confidence
    upper_bound_probability = 0.5 + half_confidence

    slope_distribution_figure = Figure()
    slope_kernel = gaussian_kde(flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE])
    slope_max = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].max()
    slope_min = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].min()
    slope_difference = slope_max - slope_min
    slope_margin = slope_difference * 0.05
    slope_plotting_positions = np.linspace(slope_min - slope_margin, slope_max + slope_margin, 500)
    slope_pdf = slope_kernel(slope_plotting_positions)
    slope_cdf_values = slope_pdf.cumsum() / slope_pdf.cumsum().max()
    slope_ppf = interp1d(slope_cdf_values, slope_plotting_positions, bounds_error=True)
    slope_lower_bound_value = float(slope_ppf(lower_bound_probability))
    slope_median_value = float(slope_ppf(0.5))
    slope_upper_bound_value = float(slope_ppf(upper_bound_probability))
    slope_deviation_plotting_positions = np.linspace(slope_lower_bound_value, slope_upper_bound_value, 500)
    slope_distribution_figure.varea(x=slope_deviation_plotting_positions, y1=0,
                                    y2=slope_kernel(slope_deviation_plotting_positions),
                                    alpha=0.3)
    slope_distribution_figure.segment(x0=slope_median_value, x1=slope_median_value, y0=0,
                                      y1=slope_kernel(slope_median_value))
    slope_distribution_figure.line(slope_plotting_positions, slope_pdf)

    intercept_distribution_figure = Figure()
    intercept_kernel = gaussian_kde(flaring_metadata_data_frame[
                                        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT])
    intercept_max = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].max()
    intercept_min = flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].min()
    intercept_difference = intercept_max - intercept_min
    intercept_margin = intercept_difference * 0.05
    intercept_plotting_positions = np.linspace(intercept_min - intercept_margin, intercept_max + intercept_margin, 500)
    intercept_pdf = intercept_kernel(intercept_plotting_positions)
    intercept_cdf_values = intercept_pdf.cumsum() / intercept_pdf.cumsum().max()
    intercept_ppf = interp1d(intercept_cdf_values, intercept_plotting_positions, bounds_error=True)
    intercept_lower_bound_value = float(intercept_ppf(lower_bound_probability))
    intercept_median_value = float(intercept_ppf(0.5))
    intercept_upper_bound_value = float(intercept_ppf(upper_bound_probability))
    intercept_deviation_plotting_positions = np.linspace(intercept_lower_bound_value, intercept_upper_bound_value, 500)
    intercept_distribution_figure.varea(x=intercept_deviation_plotting_positions, y1=0,
                                        y2=intercept_kernel(intercept_deviation_plotting_positions),
                                        alpha=0.3)
    intercept_distribution_figure.segment(x0=intercept_median_value, x1=intercept_median_value, y0=0,
                                          y1=intercept_kernel(intercept_median_value))
    intercept_distribution_figure.line(intercept_plotting_positions, intercept_pdf)

    row = Row(ffd_figure, slope_distribution_figure, intercept_distribution_figure)
    show(row)