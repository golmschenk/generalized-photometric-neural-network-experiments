import numpy as np
from backports.strenum import StrEnum
from bokeh.colors import Color
from bokeh.io import show
from bokeh.models import Row, ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure as Figure
from pathlib import Path
import tensorflow as tf
import pandas as pd

from generalized_photometric_neural_network_experiments.dataset.flare.light_curve_collection import \
    FlareExperimentLightCurveCollection
from generalized_photometric_neural_network_experiments.dataset.flare.metrics import FlareThresholdedCalculator
from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import MetadataColumnName
from ramjet.data_interface.tess_data_interface import TessDataInterface

infer_full_path = Path(
    '/Users/golmsche/Desktop/CuraWithLateAuxiliaryNoSigmoid_2021_10_06_12_34_59/infer_full.csv')
infer_train_mode_results_path = Path(
    '/Users/golmsche/Desktop/CuraWithLateAuxiliaryNoSigmoid_2021_10_06_12_34_59/infer_train_mode.csv')
infer_test_mode_results_path = Path(
    '/Users/golmsche/Desktop/CuraWithLateAuxiliaryNoSigmoid_2021_10_06_12_34_59/infer_test_mode.csv')


class FullDataColumnName(StrEnum):
    TRAIN_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE = 'train_slope_squared_scaled_thresholded_difference'
    TEST_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE = 'test_slope_squared_scaled_thresholded_difference'
    TRAIN_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE = 'train_intercept_squared_scaled_thresholded_difference'
    TEST_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE = 'test_intercept_squared_scaled_thresholded_difference'
    TEST_PREDICTED_SLOPE = 'test_predicted_slope'
    TEST_PREDICTED_INTERCEPT = 'test_predicted_intercept'


def gather_results_in_single_data_frame():
    dataframe = FlareExperimentLightCurveCollection().metadata_data_frame
    tess_data_interface = TessDataInterface()
    for mode, infer_path in {'train': infer_train_mode_results_path, 'test': infer_test_mode_results_path}.items():
        infer_data_frame = pd.read_csv(infer_path)
        infer_data_frame.rename(
            columns={'label_0_confidence': f'{mode}_predicted_slope',
                     'label_1_confidence': f'{mode}_predicted_intercept'},
            inplace=True)
        lambdafunc = lambda x: pd.Series(
            tess_data_interface.get_tic_id_and_sector_from_file_path(x['light_curve_path']))
        infer_data_frame[[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR]] = infer_data_frame.apply(
            lambdafunc, axis=1)
        dataframe = dataframe.merge(infer_data_frame, how='left',
                                    on=[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR])

    flare_thresholded_calculator = FlareThresholdedCalculator()
    for mode in ['train', 'test']:
        dataframe[[f'{mode}_slope_scaled_thresholded_absolute_difference',
                   f'{mode}_intercept_scaled_thresholded_absolute_difference']] = dataframe.apply(
            lambda row: pd.Series(flare_thresholded_calculator.scaled_thresholded_absolute_difference(
                tf.convert_to_tensor([
                    row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                    row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
                ]),
                tf.convert_to_tensor([row[f'{mode}_predicted_slope'], row[f'{mode}_predicted_intercept']])
            ).numpy()[0]), axis=1)
        dataframe[[f'{mode}_slope_squared_scaled_thresholded_difference',
                   f'{mode}_intercept_squared_scaled_thresholded_difference']] = dataframe.apply(
            lambda row: pd.Series(flare_thresholded_calculator.squared_scaled_thresholded_difference(
                tf.convert_to_tensor([
                    row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                    row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
                ]),
                tf.convert_to_tensor([row[f'{mode}_predicted_slope'], row[f'{mode}_predicted_intercept']])
            ).numpy()[0]), axis=1)

    return dataframe


def plot_analysis(full_data_frame: pd.DataFrame):
    palette = Category10[10]
    flaring_data_frame = full_data_frame[full_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].notna()]
    non_flaring_data_frame = full_data_frame[
        full_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].isna()]
    flaring_values = flaring_data_frame[FullDataColumnName.TEST_PREDICTED_INTERCEPT].values
    non_flaring_values = non_flaring_data_frame[FullDataColumnName.TEST_PREDICTED_INTERCEPT].values
    figure = Figure()
    non_flaring_histogram_values, non_flaring_bin_edges = np.histogram(non_flaring_values, bins=50,
                                                                       density=True)
    flaring_histogram_values, flaring_bin_edges = np.histogram(flaring_values, bins=50,
                                                               density=True)
    fill_alpha = 0.5
    line_alpha = 0.7
    figure.quad(top=non_flaring_histogram_values, bottom=0,
                left=non_flaring_bin_edges[:-1], right=non_flaring_bin_edges[1:],
                fill_alpha=fill_alpha, color=palette[0], legend_label='Non-flaring',
                line_alpha=line_alpha)
    figure.quad(top=flaring_histogram_values, bottom=0,
                left=flaring_bin_edges[:-1], right=flaring_bin_edges[1:],
                fill_alpha=fill_alpha, color=palette[1], legend_label='Flaring',
                line_alpha=line_alpha)
    figure.y_range.start = 0
    show(figure)


def create_light_curve_figure(fluxes0, times0, name0, title, x_axis_label='Time (BTJD)',
                              y_axis_label='Relative flux') -> Figure:
    figure = Figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, active_drag='box_zoom')

    def add_light_curve(times, fluxes, legend_label, color):
        """Adds a light curve to the figure."""
        # fluxes -= np.minimum(np.nanmin(fluxes), 0)
        # flux_median = np.median(fluxes)
        # relative_fluxes = fluxes / flux_median
        figure.line(times, fluxes, line_color=color, line_alpha=0.1)
        figure.circle(times, fluxes, legend_label=legend_label, line_color=color, line_alpha=0.4,
                      fill_color=color, fill_alpha=0.1)

    add_light_curve(times0, fluxes0, name0, 'mediumblue')
    figure.sizing_mode = 'stretch_width'
    return figure


def add_infer_results_to_figure(infer_path: Path, slope_figure: Figure, intercept_figure: Figure, color: Color) -> None:
    dataframe = FlareExperimentLightCurveCollection().metadata_data_frame
    tess_data_interface = TessDataInterface()
    infer_data_frame = pd.read_csv(infer_path)
    infer_data_frame.rename(
        columns={'label_0_confidence': f'predicted_slope',
                 'label_1_confidence': f'predicted_intercept'},
        inplace=True)
    lambdafunc = lambda x: pd.Series(
        tess_data_interface.get_tic_id_and_sector_from_file_path(x['light_curve_path']))
    infer_data_frame[[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR]] = infer_data_frame.apply(
        lambdafunc, axis=1)
    dataframe = dataframe.merge(infer_data_frame, how='inner',
                                on=[MetadataColumnName.TIC_ID, MetadataColumnName.SECTOR])
    column_data_source = ColumnDataSource(data=dataframe)
    true_slope_minimum = dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].min()
    true_slope_maximum = dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].max()
    from sklearn.metrics import r2_score
    slope_coefficient_of_determination = r2_score(dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                                                  dataframe[f'predicted_slope'])
    slope_figure.circle(source=column_data_source, x=MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE,
                        y=f'predicted_slope', color=color)
    slope_figure.line(x=[true_slope_minimum, true_slope_maximum], y=[true_slope_minimum, true_slope_maximum],
                      color='black')
    true_intercept_minimum = dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].min()
    true_intercept_maximum = dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT].max()
    intercept_coefficient_of_determination = r2_score(
        dataframe[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT],
        dataframe[f'predicted_intercept'])
    intercept_figure.circle(source=column_data_source, x=MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT,
                            y=f'predicted_intercept', color=color)
    intercept_figure.line(x=[true_intercept_minimum, true_intercept_maximum],
                          y=[true_intercept_minimum, true_intercept_maximum],
                          color='black')


if __name__ == '__main__':
    slope_title = f'Slope'
    # slope_title += f' (R^2={slope_coefficient_of_determination})'
    slope_figure_ = Figure(match_aspect=True, title=slope_title,
                           x_axis_label='True', y_axis_label='Predicted')
    intercept_title = f'Intercept'
    # intercept_title += ' (R^2={intercept_coefficient_of_determination})'
    intercept_figure_ = Figure(match_aspect=True, title=intercept_title,
                               x_axis_label='True', y_axis_label='Predicted')
    infer_path_ = Path('logs/HadesWithFlareInterceptLuminosityAddedNoSigmoid_luminosity_as_linear_2021_10_30_11_51_40/infer_results_2021-11-07-18-46-54.csv')
    add_infer_results_to_figure(infer_path_, slope_figure_, intercept_figure_, color=Category10[9][0])
    infer_path_ = Path('logs/HadesWithFlareInterceptLuminosityAddedNoSigmoid_plain_2021_11_11_14_29_11/infer_results_2021-11-15-10-06-54.csv')
    add_infer_results_to_figure(infer_path_, slope_figure_, intercept_figure_, color=Category10[9][1])
    show(Row(slope_figure_, intercept_figure_))
