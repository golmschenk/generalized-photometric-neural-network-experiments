import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from altaipony.fakeflares import aflare
from bokeh.io import show
from bokeh.models import Column, Row
from bokeh.palettes import Blues, Category10
from bokeh.plotting import Figure
from bs4 import BeautifulSoup
from astropy.io import ascii
import scipy.stats

from generalized_photometric_neural_network_experiments.dataset.flare.flare_frequency_distribution_math import \
    convert_flare_frequency_distribution_from_absolute_to_equivalent_duration
from generalized_photometric_neural_network_experiments.dataset.flare.names_and_paths import MetadataColumnName


def generate_flare_time_comparison():
    times0 = np.linspace(-1, 1, 1000)
    fluxes0 = aflare(t=times0, tpeak=0, dur=0.1, ampl=1, upsample=True)
    figure0 = Figure()
    figure0.line(x=times0, y=fluxes0, line_alpha=0.1)
    figure0.circle(x=times0, y=fluxes0, line_alpha=0.4, fill_alpha=0.1)
    times1 = np.linspace(-500, 500, 1000)
    fluxes1 = aflare(t=times1, tpeak=0, dur=400, ampl=1, upsample=True)
    figure1 = Figure()
    figure1.line(x=times1, y=fluxes1, line_alpha=0.1)
    figure1.circle(x=times1, y=fluxes1, line_alpha=0.4, fill_alpha=0.1)
    show(Column(figure0, figure1))


def download_maximilian_gunther_flare_data_frame() -> pd.DataFrame:
    """
    Gets the relevant metadata from the flare catalog paper by Maximilian Gunther et al.
    https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a

    :return: The data frame of the TIC IDs and flare statistics.
    """
    request_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
                                     '(KHTML, like Gecko) Version/15.1 Safari/605.1.15'}  # Prevent bot blocking.
    paper_page_response = requests.get('https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a',
                                       headers=request_headers)
    paper_page_soup = BeautifulSoup(paper_page_response.content, 'html.parser')
    if 'captcha' in paper_page_soup.find('title').text.lower():
        raise ConnectionError('The journal page (correctly) thinks this is a bot. Manually visit '
                              'https://iopscience.iop.org/article/10.3847/1538-3881/ab5d3a then try again.')
    table_data_link_element = paper_page_soup.find(class_='wd-jnl-art-btn-table-data')
    paper_data_table_url = table_data_link_element['href']
    paper_data_table_response = requests.get(paper_data_table_url)
    paper_data_table = ascii.read(io.BytesIO(paper_data_table_response.content))
    paper_data_frame = paper_data_table.to_pandas()
    assert paper_data_frame.shape[0] > 0
    assert paper_data_frame['Amp'].notna().all()
    assert paper_data_frame['FWHM'].notna().all()
    return paper_data_frame


def plot_gunther_flare_amplitude_and_duration_kde_with_resample():
    gunther_flare_data_frame = download_maximilian_gunther_flare_data_frame()
    log_amplitude_column = np.log10(gunther_flare_data_frame['Amp'])
    log_full_width_at_half_maximum_column = np.log10(gunther_flare_data_frame['FWHM'])
    kernel = scipy.stats.gaussian_kde(np.stack([log_amplitude_column.values,
                                                log_full_width_at_half_maximum_column.values], axis=0))
    resample = kernel.resample(9000)
    resampled_log_full_width_at_half_maximums = resample[1, :]
    resampled_log_amplitudes = resample[0, :]
    minimum_amplitude = np.min(np.concatenate([log_amplitude_column.values, resampled_log_amplitudes]))
    maximum_amplitude = np.max(np.concatenate([log_amplitude_column.values, resampled_log_amplitudes]))
    minimum_full_width_at_half_maximum = np.min(np.concatenate([log_full_width_at_half_maximum_column.values,
                                                                resampled_log_full_width_at_half_maximums]))
    maximum_full_width_at_half_maximum = np.max(np.concatenate([log_full_width_at_half_maximum_column.values,
                                                                resampled_log_full_width_at_half_maximums]))
    image_size = 500
    y_step_size = (maximum_amplitude - minimum_amplitude) / image_size
    x_step_size = (maximum_full_width_at_half_maximum - minimum_full_width_at_half_maximum) / image_size
    epsilon = 1e-6
    xy = np.mgrid[minimum_amplitude:maximum_amplitude - epsilon:y_step_size,
         minimum_full_width_at_half_maximum: maximum_full_width_at_half_maximum - epsilon:x_step_size,
         ].reshape(2, -1)
    densities = kernel(xy)
    density_map = densities.reshape(image_size, image_size)
    figure = Figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                    title='Flare properties', x_axis_label='Full width at half maximum (days)',
                    y_axis_label='Relative amplitude')
    figure.x_range.range_padding = figure.y_range.range_padding = 0
    figure.image(image=[density_map],
                 x=minimum_full_width_at_half_maximum,
                 y=minimum_amplitude,
                 dw=maximum_full_width_at_half_maximum - minimum_full_width_at_half_maximum,
                 dh=maximum_amplitude - minimum_amplitude,
                 palette=Blues[9][::-1], level="image")
    figure.grid.grid_line_width = 0.5
    figure.circle(x=log_full_width_at_half_maximum_column, y=log_amplitude_column, color=Category10[10][1],
                  alpha=0.7, size=1)
    figure.circle(x=resampled_log_full_width_at_half_maximums, y=resampled_log_amplitudes, color=Category10[10][2],
                  alpha=0.7, size=1)
    show(figure)


def generate_gunther_based_flares():
    gunther_flare_data_frame = download_maximilian_gunther_flare_data_frame()
    log_amplitude_column = np.log10(gunther_flare_data_frame['Amp'])
    log_full_width_at_half_maximum_column = np.log10(gunther_flare_data_frame['FWHM'])
    kernel = scipy.stats.gaussian_kde(np.stack([log_amplitude_column.values,
                                                log_full_width_at_half_maximum_column.values], axis=0))
    resample = kernel.resample(100_000, seed=0)
    resampled_log_amplitudes = resample[0, :]
    resampled_log_full_width_at_half_maximums = resample[1, :]
    synthetic_flares_directory = Path('data/flare/synthetic_flares')
    synthetic_flares_directory.mkdir(exist_ok=True)
    synthetic_flare_metadata_dictionary = {'file_name': [], 'equivalent_duration': [], 'amplitude': [],
                                           'full_width_half_maximum__days': []}
    for index, (log_amplitude, log_full_width_at_half_maximum) in enumerate(
            zip(resampled_log_amplitudes, resampled_log_full_width_at_half_maximums)):
        amplitude = 10 ** log_amplitude
        full_width_at_half_maximum = 10 ** log_full_width_at_half_maximum
        flare_times, flare_fluxes = generate_flare_for_amplitude_and_full_width_at_half_maximum(
            amplitude, full_width_at_half_maximum)
        flare_data_frame = pd.DataFrame({'flux': flare_fluxes, 'time__days': flare_times})
        flare_data_frame.to_feather(str(synthetic_flares_directory.joinpath(f'{index}.feather')))
        synthetic_flare_metadata_dictionary['file_name'].append(f'{index}.feather')
        synthetic_flare_metadata_dictionary['equivalent_duration'].append(np.trapz(y=flare_fluxes, x=flare_times))
        synthetic_flare_metadata_dictionary['amplitude'].append(amplitude)
        synthetic_flare_metadata_dictionary['full_width_half_maximum__days'].append(full_width_at_half_maximum)
    synthetic_flare_metadata_data_frame = pd.DataFrame(synthetic_flare_metadata_dictionary)
    synthetic_flare_metadata_data_frame.to_csv('data/flare/synthetic_flare_metadata.csv', index=False)


def generate_flare_for_amplitude_and_full_width_at_half_maximum(amplitude: float, full_width_at_half_maximum: float,
                                                                time_step: float = 2.778e-4
                                                                ) -> (np.ndarray, np.ndarray):
    times = np.arange(-10, 10, time_step)
    fluxes = aflare(t=times, tpeak=0, dur=full_width_at_half_maximum * 2, ampl=amplitude, upsample=False)
    indexes_of_interest = np.argwhere(fluxes > 0.001 * amplitude)
    first_index_of_interest = indexes_of_interest[0][0]
    last_index_of_interest = indexes_of_interest[-1][0]
    assert first_index_of_interest != 0
    assert last_index_of_interest != fluxes.shape[0] - 1
    times_of_interest = times[first_index_of_interest:last_index_of_interest + 1]
    fluxes_of_interest = fluxes[first_index_of_interest:last_index_of_interest + 1]
    return times_of_interest, fluxes_of_interest


def plot_synthetic_flare_equivalent_distribution_histogram():
    synthetic_flare_metadata_path = Path('data/flare/synthetic_flare_metadata.csv')
    synthetic_flare_metadata_data_frame = pd.read_csv(synthetic_flare_metadata_path, index_col=None)
    linear_bin_values, linear_bin_edges = np.histogram(
        synthetic_flare_metadata_data_frame['equivalent_duration'].values, bins=50)
    linear_figure = Figure(title='Flare equivalent duration')
    linear_figure.quad(top=linear_bin_values, bottom=0, left=linear_bin_edges[:-1], right=linear_bin_edges[1:],
                       fill_color="navy", line_color="white", alpha=0.5)
    log_bin_values, log_bin_edges = np.histogram(
        np.log10(synthetic_flare_metadata_data_frame['equivalent_duration'].values), bins=50)
    log_figure = Figure(title='Log flare equivalent duration')
    log_figure.quad(top=log_bin_values, bottom=0, left=log_bin_edges[:-1], right=log_bin_edges[1:],
                    fill_color="navy", line_color="white", alpha=0.5)
    show(Row(linear_figure, log_figure))


def plot_metadata_flare_frequency_distributions_in_equivalent_duration():
    metadata_data_frame = pd.read_csv('data/flare/metadata.csv')
    flaring_metadata_data_frame = metadata_data_frame.dropna()
    unique_flaring_metadata_data_frame = flaring_metadata_data_frame.drop_duplicates(subset=[
        MetadataColumnName.TIC_ID])
    unique_flaring_metadata_data_frame[
        MetadataColumnName.EQUIVALENT_DURATION_FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT] = \
        unique_flaring_metadata_data_frame.apply(
            lambda data_frame: convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(
                slope=data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE],
                intercept=data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT],
                log_luminosity=data_frame[MetadataColumnName.LUMINOSITY__LOG_10_SOLAR_UNITS])[1], axis=1)

    ed_slopes = unique_flaring_metadata_data_frame[MetadataColumnName.FLARE_FREQUENCY_DISTRIBUTION_SLOPE]
    ed_intercepts = unique_flaring_metadata_data_frame[
        MetadataColumnName.EQUIVALENT_DURATION_FLARE_FREQUENCY_DISTRIBUTION_INTERCEPT]
    kernel = scipy.stats.gaussian_kde(np.stack([ed_slopes.values, ed_intercepts.values], axis=0))
    minimum_slope = ed_slopes.min()
    maximum_slope = ed_slopes.max()
    minimum_intercept = ed_intercepts.min()
    maximum_intercept = ed_intercepts.max()
    image_size = 2000
    y_step_size = (maximum_slope - minimum_slope) / image_size
    x_step_size = (maximum_intercept - minimum_intercept) / image_size
    epsilon = 1e-6
    xy = np.mgrid[minimum_slope:maximum_slope - epsilon:y_step_size,
         minimum_intercept: maximum_intercept - epsilon:x_step_size].reshape(2, -1)
    densities = kernel(xy)
    density_map = densities.reshape(image_size, image_size)
    figure = Figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                    title='Equivalent duration flare frequency distributions',
                    x_axis_label='Intercept', y_axis_label='Slope')
    figure.x_range.range_padding = figure.y_range.range_padding = 0
    figure.image(image=[density_map],
                 x=minimum_intercept,
                 y=minimum_slope,
                 dw=maximum_intercept - minimum_intercept,
                 dh=maximum_slope - minimum_slope,
                 palette=Blues[9][::-1], level="image")
    figure.grid.grid_line_width = 0.5
    figure.circle(x=ed_intercepts, y=ed_slopes, color=Category10[10][1],
                  alpha=0.7, size=2)
    show(figure)


if __name__ == '__main__':
    generate_flare_time_comparison()
