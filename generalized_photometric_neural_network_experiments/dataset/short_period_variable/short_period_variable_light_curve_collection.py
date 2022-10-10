import math
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.signal

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


class MixedSineAndSawtoothWaveLightCurveCollection(TessFfiLightCurveCollection):
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
        sine_amplitude = np.random.uniform(0, amplitude)
        sawtooth_amplitude = amplitude - sine_amplitude        
        sine_phases = np.linspace(0, math.tau * periods_to_produce, num=light_curve_length, endpoint=False)
        sine_magnifications = (np.sin(sine_phases) * sine_amplitude)
        sawtooth_phase_offset = np.random.uniform(0, math.tau)
        sawtooth_rising_ramp_width = np.random.uniform(0, 1)
        sawtooth_phases = np.linspace(sawtooth_phase_offset, (math.tau * periods_to_produce) + sawtooth_phase_offset,
                                      num=light_curve_length, endpoint=False)
        sawtooth_magnifications = (scipy.signal.sawtooth(sawtooth_phases, width=sawtooth_rising_ramp_width)
                                   * sawtooth_amplitude)
        magnifications = sine_magnifications + sawtooth_magnifications + 1
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
