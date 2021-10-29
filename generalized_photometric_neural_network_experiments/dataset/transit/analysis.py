import numpy as np

from generalized_photometric_neural_network_experiments.dataset.transit.light_curve_collection import \
    TransitExperimentLightCurveCollection


def show_dataset_statistics():
    collection = TransitExperimentLightCurveCollection()
    light_curve_paths = collection.get_paths()
    print(f'Number of light curves: {len(list(light_curve_paths))}')
    light_curve_lengths = []
    for light_curve_path in light_curve_paths:
        times, fluxes = collection.load_times_and_fluxes_from_path(light_curve_path)
        light_curve_lengths.append(times.shape[0])
    print(f'Mean light curve length: {np.mean(light_curve_lengths)}')
    print(f'Median light curve length: {np.median(light_curve_lengths)}')


if __name__ == '__main__':
    show_dataset_statistics()
