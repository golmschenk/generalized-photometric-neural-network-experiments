import pytest

from generalized_photometric_neural_network_experiments.dataset.flare.flare_frequency_distribution_math import \
    convert_flare_frequency_distribution_from_equivalent_duration_to_absolute, \
    convert_flare_frequency_distribution_from_absolute_to_equivalent_duration, \
    convert_equivalent_duration_in_days_to_ergs


def test_convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute():
    input_slope0 = -0.5
    input_intercept0 = 2
    input_log_luminosity0 = 0
    output_slope0, output_intercept0 = (
        convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(slope=input_slope0,
                                                                                  intercept=input_intercept0,
                                                                                  log_luminosity=input_log_luminosity0))
    assert output_slope0 == -0.5
    assert output_intercept0 == 2
    input_slope1 = -0.5
    input_intercept1 = 2
    input_log_luminosity1 = 2
    output_slope1, output_intercept1 = (
        convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(slope=input_slope1,
                                                                                  intercept=input_intercept1,
                                                                                  log_luminosity=input_log_luminosity1))
    assert output_slope1 == -0.5
    assert output_intercept1 == 3
    input_slope2 = -0.5
    input_intercept2 = 2
    input_log_luminosity2 = -1
    output_slope2, output_intercept2 = (
        convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(slope=input_slope2,
                                                                                  intercept=input_intercept2,
                                                                                  log_luminosity=input_log_luminosity2))
    assert output_slope2 == -0.5
    assert output_intercept2 == 1.5
    input_slope3 = -3
    input_intercept3 = 2
    input_log_luminosity3 = 1
    output_slope3, output_intercept3 = (
        convert_flare_frequency_distribution_from_equivalent_duration_to_absolute(slope=input_slope3,
                                                                                  intercept=input_intercept3,
                                                                                  log_luminosity=input_log_luminosity3))
    assert output_slope3 == -3
    assert output_intercept3 == 5


def test_convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration():
    input_slope0 = -0.5
    input_intercept0 = 2
    input_log_luminosity0 = 0
    output_slope0, output_intercept0 = (
        convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(slope=input_slope0,
                                                                                  intercept=input_intercept0,
                                                                                  log_luminosity=input_log_luminosity0))
    assert output_slope0 == -0.5
    assert output_intercept0 == 2
    input_slope1 = -0.5
    input_intercept1 = 3
    input_log_luminosity1 = 2
    output_slope1, output_intercept1 = (
        convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(slope=input_slope1,
                                                                                  intercept=input_intercept1,
                                                                                  log_luminosity=input_log_luminosity1))
    assert output_slope1 == -0.5
    assert output_intercept1 == 2
    input_slope2 = -0.5
    input_intercept2 = 1.5
    input_log_luminosity2 = -1
    output_slope2, output_intercept2 = (
        convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(slope=input_slope2,
                                                                                  intercept=input_intercept2,
                                                                                  log_luminosity=input_log_luminosity2))
    assert output_slope2 == -0.5
    assert output_intercept2 == 2
    input_slope3 = -3
    input_intercept3 = 5
    input_log_luminosity3 = 1
    output_slope3, output_intercept3 = (
        convert_flare_frequency_distribution_from_absolute_to_equivalent_duration(slope=input_slope3,
                                                                                  intercept=input_intercept3,
                                                                                  log_luminosity=input_log_luminosity3))
    assert output_slope3 == -3
    assert output_intercept3 == 2


def test_convert_equivalent_duration_to_ergs():
    equivalent_duration0__days = 1
    energy__ergs0 = convert_equivalent_duration_in_days_to_ergs(equivalent_duration__days=equivalent_duration0__days)
    assert energy__ergs0 == pytest.approx(3.305664e38)
    equivalent_duration1__days = 0.1
    star_luminosity1__solar_units = 0.5
    energy__ergs1 = convert_equivalent_duration_in_days_to_ergs(
        equivalent_duration__days=equivalent_duration1__days,
        star_luminosity__solar_units=star_luminosity1__solar_units)
    assert energy__ergs1 == pytest.approx(1.652832e37)
