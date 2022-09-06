import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection


class SineWaveLightCurveCollection(TessFfiLightCurveCollection):
    def __init__(self, min_period__days: float, max_period__days: float):
        super().__init__()
        self.min_period__days: float = min_period__days
        self.max_period__days: float = max_period__days

    def get_paths(self) -> Iterable[Path]:
        return [Path('')]

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a random generated signal.

        :param path: empty path
        :return: The times and the magnifications of the signal.
        """

        period__days = np.random.uniform(low=self.min_period__days, high=self.max_period__days)
        length__days = 30
        light_curve_length = length__days * 24 * 4  # 15 minute intervals.
        periods_to_produce = length__days / period__days
        max_amplitude = 1
        min_amplitude = 0.001
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        phase = np.linspace(0, math.tau * periods_to_produce, num=light_curve_length, endpoint=False)
        magnifications = (np.sin(phase) * amplitude) + 1
        times = np.linspace(0, 30, num=light_curve_length, endpoint=False)
        return times, magnifications


class UniformNoiseLightCurveCollection(TessFfiLightCurveCollection):
    def get_paths(self) -> Iterable[Path]:
        return [Path('')]

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a random generated signal.

        :param path: empty path
        :return: The times and the magnifications of the signal.
        """
        length__days = 30
        light_curve_length = length__days * 24 * 4  # 15 minute intervals.
        max_amplitude = 1
        min_amplitude = 0.001
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        times = np.linspace(0, 30, num=light_curve_length, endpoint=False)
        magnifications = np.random.uniform(low=1 - amplitude, high=1 + amplitude, size=times.shape)
        return times, magnifications
