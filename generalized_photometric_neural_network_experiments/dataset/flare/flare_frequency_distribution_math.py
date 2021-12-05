def convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration(
        log_slope: float, log_intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = log_intercept + (log_slope * log_luminosity)
    return log_slope, shifted_log_intercept


def convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute(
        log_slope: float, log_intercept: float, log_luminosity: float) -> (float, float):
    shifted_log_intercept = log_intercept - (log_slope * log_luminosity)
    return log_slope, shifted_log_intercept
