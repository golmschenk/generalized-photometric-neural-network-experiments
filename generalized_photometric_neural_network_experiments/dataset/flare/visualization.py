"""
Visualization of the flare dataset.
"""
from collections import namedtuple
from dataclasses import dataclass

from bokeh.transform import factor_cmap
from typing import List, Optional

from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import show, save
from bokeh.models import ColumnDataSource, PrintfTickFormatter, Row, FactorRange, Column, LabelSet
from bokeh.palettes import Category10, Set2, Set1, Dark2, Accent, Category20
from bokeh.plotting import Figure
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import MetadataColumnName
from ramjet.data_interface.tess_data_interface import TessDataInterface


def show_flare_frequency_distribution_plots() -> None:
    """
    Show some plots about the flare frequency distribution statistics.
    """
    metadata_data_frame = pd.read_csv('data/flare/metadata.csv')
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


def download_luminosity_distribution(use_existing_csv: bool = False) -> None:
    metadata_csv_path = Path('metadata_with_extra.csv')

    @dataclass
    class Field:
        name: str
        tic_name: str
        plural_name: str
        axis_name: str
        log_scale: Optional[bool] = True
    fields: List[Field] = [
        Field('luminosity', 'lum', 'luminosities', 'luminosity (solar units)'),
        Field('distance', 'd', 'distances', 'distance (parsec)'),
        Field('tess_magnitude', 'Tmag', 'tess_magnitudes', 'TESS magnitude', False),
        Field('mass', 'mass', 'masses', 'mass (solar units)'),
        Field('radius', 'rad', 'radii', 'radius (solar units)'),
    ]
    if not use_existing_csv:
        tess_data_interface = TessDataInterface()
        metadata_data_frame = pd.read_csv('data/flare/metadata.csv')
        for value_name in fields:
            metadata_data_frame[value_name.name] = np.nan
            for index, metadata_row in metadata_data_frame.iterrows():
                tic_id = metadata_row[MetadataColumnName.TIC_ID]
                tic_row = tess_data_interface.get_tess_input_catalog_row(tic_id)
                luminosity = tic_row[value_name.tic_name]
                metadata_data_frame.loc[index, value_name.name] = luminosity
            assert metadata_data_frame[metadata_data_frame[value_name.name] == -1.0].shape[0] == 0
        metadata_data_frame[metadata_data_frame['luminosity'] == 0.0] = np.nan
        metadata_data_frame[metadata_data_frame['distance'] == 0.0] = np.nan
        metadata_data_frame[metadata_data_frame['mass'] == 0.0] = np.nan
        metadata_data_frame[metadata_data_frame['radius'] == 0.0] = np.nan
        metadata_data_frame.to_csv(metadata_csv_path)
    metadata_data_frame = pd.read_csv(metadata_csv_path)

    flaring_data_frame = metadata_data_frame[metadata_data_frame[
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].notna()]
    all_tess_sample_data_frame = metadata_data_frame[metadata_data_frame[
        MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].isna()]

    rows = []
    for field_index, field in enumerate(fields):
        number_of_colors = 20
        palette = Category20[number_of_colors]
        all_tess_color = palette[(field_index * 2 + 1) % number_of_colors]
        flaring_color = palette[(field_index * 2 + 4) % number_of_colors]
        all_tess_sample_field_values = all_tess_sample_data_frame[
            all_tess_sample_data_frame[field.name].notna()][field.name].values
        flaring_field_values = flaring_data_frame[flaring_data_frame[field.name].notna()][field.name].values
        if field.log_scale:
            axis_type = 'log'
            scaled_all_tess_sample_field_values = np.log10(all_tess_sample_field_values)
            scaled_flaring_field_values = np.log10(flaring_field_values)
        else:
            axis_type = 'linear'
            scaled_all_tess_sample_field_values = all_tess_sample_field_values
            scaled_flaring_field_values = flaring_field_values
        field_histogram_figure = Figure(x_axis_type=axis_type, title=f'{field.plural_name} histogram'.capitalize(),
                                        x_axis_label=f'{field.axis_name}'.capitalize(), y_axis_label='Density')
        histogram_values, bin_edges = np.histogram(scaled_all_tess_sample_field_values, bins=50,
                                                   density=True)
        flaring_histogram_values, flaring_bin_edges = np.histogram(scaled_flaring_field_values, bins=50,
                                                                   density=True,
                                                                   range=(bin_edges[0], bin_edges[-1]))
        flaring_histogram_values *= (len(flaring_field_values) / len(all_tess_sample_field_values))
        if field.log_scale:
            scaled_bin_edges = 10 ** bin_edges
        else:
            scaled_bin_edges = bin_edges
        fill_alpha = 0.8
        line_alpha = 0.9
        field_histogram_figure.quad(top=histogram_values, bottom=0,
                                    left=scaled_bin_edges[:-1], right=scaled_bin_edges[1:],
                                    fill_alpha=fill_alpha, color=all_tess_color, legend_label='All TESS',
                                    line_alpha=line_alpha)
        field_histogram_figure.quad(top=flaring_histogram_values, bottom=0,
                                    left=scaled_bin_edges[:-1], right=scaled_bin_edges[1:],
                                    fill_alpha=fill_alpha, color=flaring_color, legend_label='Known flaring',
                                    line_alpha=line_alpha)
        field_histogram_figure.y_range.start = 0

        all_tess_sample_has_field_percentage = (
                all_tess_sample_data_frame[field.name].notna().sum() / all_tess_sample_data_frame.shape[0])
        all_tess_sample_nan_field_percentage = (
                all_tess_sample_data_frame[field.name].isna().sum() / all_tess_sample_data_frame.shape[0])
        flaring_has_field_count = flaring_data_frame[field.name].notna().sum() / flaring_data_frame.shape[0]
        flaring_nan_field_count = flaring_data_frame[field.name].isna().sum() / flaring_data_frame.shape[0]
        values = [('Exists', 'All TESS'), ('Exists', 'Known flaring'), ('NaN', 'All TESS'), ('NaN', 'Known flaring')]
        field_existence_figure = Figure(x_range=FactorRange(*values), title=f'Light curves with {field.name} value',
                                        x_axis_label='Value', y_axis_label='Percentage')
        legend_names = [level1 for level0, level1 in values]
        percentages = [all_tess_sample_has_field_percentage, flaring_has_field_count,
                       all_tess_sample_nan_field_percentage, flaring_nan_field_count]
        field_existence_column_data_source = ColumnDataSource(
            data={'value': values,
                  'percentage': percentages,
                  'legend_name': legend_names,
                  'rounded_percentage': np.around(percentages, decimals=2)
                  }
        )
        bar_color_map = factor_cmap('value', palette=[all_tess_color, flaring_color], factors=['All TESS', 'Known flaring'],
                                    start=1, end=2)
        field_existence_figure.vbar(x='value', top='percentage', source=field_existence_column_data_source, width=0.9,
                                    fill_alpha=fill_alpha, color=bar_color_map, line_alpha=line_alpha,
                                    legend_field='legend_name')
        percentage_labels = LabelSet(x='value', y='percentage', text='rounded_percentage', level='glyph',
                                     x_offset=-13.5, y_offset=0, source=field_existence_column_data_source,
                                     render_mode='canvas')
        field_existence_figure.add_layout(percentage_labels)
        field_existence_figure.x_range.range_padding = 0.2
        field_existence_figure.x_range.group_padding = 0.4
        field_existence_figure.y_range.start = 0
        field_existence_figure.y_range.end = 1
        row = Row(
            field_histogram_figure,
            field_existence_figure)
        rows.append(row)
    column = Column(*rows)
    show(column)


if __name__ == '__main__':
    download_luminosity_distribution(use_existing_csv=True)
    # show_flare_frequency_distribution_plots()