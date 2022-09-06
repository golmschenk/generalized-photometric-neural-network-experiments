import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ramjet.photometric_database.derived.tess_ffi_light_curve_collection import TessFfiLightCurveCollection


class SyntheticShortPeriodLightCurveCollection(TessFfiLightCurveCollection):
    def get_paths(self) -> Iterable[Path]:
        return [Path('')]

    def load_times_and_magnifications_from_path(self, path: Path) -> (np.ndarray, np.ndarray):
        """
        Loads the times and magnifications from a random generated signal.

        :param path: empty path
        :return: The times and the magnifications of the signal.
        """
        max_period__hours = 5
        min_period__hours = 0.25
        max_period__days = max_period__hours / 24
        min_period__days = min_period__hours / 24
        period__days = np.random.uniform(low=min_period__days, high=max_period__days)
        length__days = 30
        light_curve_length = length__days * 24 * 4  # 15 minute intervals.
        periods_to_produce = length__days / period__days
        max_amplitude = 1
        min_amplitude = 0.001
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        times = np.linspace(0, math.tau * periods_to_produce, num=light_curve_length, endpoint=False)
        magnifications = (np.sin(times) * amplitude) + 1
        return times, magnifications
