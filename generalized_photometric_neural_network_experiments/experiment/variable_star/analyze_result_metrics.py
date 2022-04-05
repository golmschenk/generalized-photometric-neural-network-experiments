from pathlib import Path

import numpy as np
from bokeh.io import show
from bokeh.plotting import Figure


def show_cumulative_distributions_figure():
    cumulative_distribution_figure = Figure()
    for distribution_path in Path('.').glob('*.npy'):
        confidence_bin_counts = np.load(str(distribution_path)).astype(np.float32)
        confidence_bin_densities = confidence_bin_counts / confidence_bin_counts.sum()
        confidence_bin_cumulative_densities = np.cumsum(confidence_bin_densities)
        confidence_bin_cumulative_densities = np.concatenate([[0], confidence_bin_cumulative_densities, [1]])
        confidence_bin_values = (np.arange(10000, dtype=np.float32) + 0.5) / 10000
        confidence_bin_values = np.concatenate([[0], confidence_bin_values, [1]])
        cumulative_distribution_figure.step(confidence_bin_values, confidence_bin_cumulative_densities)
    show(cumulative_distribution_figure)


def find_index_for_cumulative_density(cumulative_densities: np.ndarray, cumulative_density_to_find: float) -> float:
    return np.searchsorted(cumulative_densities, cumulative_density_to_find, side='right')


def main():
    non_rr_lyrae_variable_bin_counts = np.load('known_non_rr_lyrae_variable_infer_results_2022-03-06-12-08-57.npy'
                                               ).astype(np.float32)
    non_rr_lyrae_variable_densities = non_rr_lyrae_variable_bin_counts / non_rr_lyrae_variable_bin_counts.sum()
    non_rr_lyrae_variable_cumulative_densities = np.cumsum(non_rr_lyrae_variable_densities)
    rr_lyrae_bin_counts = np.load('known_rr_lyrae_infer_results_2022-03-06-12-07-20.npy').astype(np.float32)
    rr_lyrae_densities = rr_lyrae_bin_counts / rr_lyrae_bin_counts.sum()
    rr_lyrae_cumulative_densities = np.cumsum(rr_lyrae_densities)
    confidence_bin_values = (np.arange(10000, dtype=np.float32) + 0.5) / 10000
    print_confidence_threshold(confidence_bin_values, non_rr_lyrae_variable_cumulative_densities,
                               rr_lyrae_cumulative_densities, 0.95)
    print_confidence_threshold(confidence_bin_values, non_rr_lyrae_variable_cumulative_densities,
                               rr_lyrae_cumulative_densities, 0.99)
    print_confidence_threshold(confidence_bin_values, non_rr_lyrae_variable_cumulative_densities,
                               rr_lyrae_cumulative_densities, 0.999)
    print_confidence_threshold(confidence_bin_values, non_rr_lyrae_variable_cumulative_densities,
                               rr_lyrae_cumulative_densities, 0.9995)
    print_confidence_threshold(confidence_bin_values, non_rr_lyrae_variable_cumulative_densities,
                               rr_lyrae_cumulative_densities, 0.9999)


def print_confidence_threshold(confidence_bin_values: np.ndarray,
                               non_rr_lyrae_variable_cumulative_densities: np.ndarray,
                               rr_lyrae_cumulative_densities: np.ndarray,
                               target_non_rr_lyrae_density: float):
    index = find_index_for_cumulative_density(non_rr_lyrae_variable_cumulative_densities, target_non_rr_lyrae_density)
    print(f'=' * 30)
    print(f'non_rr_lyrae_variable_cumulative_density: {non_rr_lyrae_variable_cumulative_densities[index]}')
    print(f'rr_lyrae_cumulative_density: {rr_lyrae_cumulative_densities[index]}')
    print(f'confidence: {confidence_bin_values[index]}')


if __name__ == '__main__':
    main()
    show_cumulative_distributions_figure()