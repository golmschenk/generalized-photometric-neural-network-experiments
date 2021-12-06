def convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(
        slope: float, intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = intercept + (slope * log_luminosity)
    return slope, shifted_log_intercept


def convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(
        slope: float, intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = intercept - (slope * log_luminosity)
    return slope, shifted_log_intercept
