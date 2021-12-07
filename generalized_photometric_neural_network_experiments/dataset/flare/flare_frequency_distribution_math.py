
solar_luminosity__ergs_per_second = 3.826e33
seconds_per_day = 86400
solar_luminosity__ergs_per_day = solar_luminosity__ergs_per_second * seconds_per_day


def convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(
        slope: float, intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = intercept + (slope * log_luminosity)
    return slope, shifted_log_intercept


def convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(
        slope: float, intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = intercept - (slope * log_luminosity)
    return slope, shifted_log_intercept


def convert_equivalent_duration_in_days_to_ergs(equivalent_duration__days: float,
                                                star_luminosity__solar_units: float = 1
                                                ) -> float:
    return equivalent_duration__days * star_luminosity__solar_units * solar_luminosity__ergs_per_day
