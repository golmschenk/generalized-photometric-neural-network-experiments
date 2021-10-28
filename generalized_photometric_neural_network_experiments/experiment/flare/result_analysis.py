import numpy as np
from backports.strenum import StrEnum
from bokeh.io import show
from bokeh.models import Row
from bokeh.palettes import Category10
from bokeh.plotting import Figure
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


def plot_analysis(data_frame: pd.DataFrame):
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


if __name__ == '__main__':
    # full_data_frame = gather_results_in_single_data_frame()
    # full_data_frame.to_csv(infer_full_path)
    full_data_frame = pd.read_csv(infer_full_path)
    # plot_analysis(full_data_frame)
    collection = FlareExperimentLightCurveCollection()
    flaring_data_frame = full_data_frame[full_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].notna()]
    flaring_data_frame = flaring_data_frame.sort_values(MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT, ascending=False)
    for index in range(flaring_data_frame.shape[0]):
        most_row = flaring_data_frame.iloc[index]
        most_path = most_row['light_curve_path_x']
        times, fluxes = collection.load_times_and_fluxes_from_path(most_path)
        title = f'{most_path}\nLum:{most_row[MetadataColumnName.LUMINOSITY__LOG_10_SOLAR_UNITS]}\nInt:{most_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]}'
        most_figure = create_light_curve_figure(fluxes, times, '', title)
        least_row = flaring_data_frame.iloc[(-1) - index]
        least_path = least_row['light_curve_path_x']
        times, fluxes = collection.load_times_and_fluxes_from_path(least_path)
        title = f'{least_path}\nLum:{least_row[MetadataColumnName.LUMINOSITY__LOG_10_SOLAR_UNITS]}\nInt:{least_row[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]}'
        least_figure = create_light_curve_figure(fluxes, times, '', title)
        show(Row(most_figure, least_figure))
        pass
    # flaring_data_frame = full_data_frame[full_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].notna()]
    # non_flaring_data_frame = full_data_frame[full_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE].isna()]
    # print(flaring_data_frame[FullDataColumnName.TEST_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(non_flaring_data_frame[FullDataColumnName.TEST_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(flaring_data_frame[FullDataColumnName.TEST_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(non_flaring_data_frame[FullDataColumnName.TEST_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(full_data_frame[FullDataColumnName.TRAIN_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(full_data_frame[FullDataColumnName.TEST_SLOPE_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(full_data_frame[FullDataColumnName.TRAIN_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
    # print(full_data_frame[FullDataColumnName.TEST_INTERCEPT_SQUARED_SCALED_THRESHOLDED_DIFFERENCE].mean())
