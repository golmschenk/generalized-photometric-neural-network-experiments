from generalized_photometric_neural_network_experiments.dataset.flare.flare_frequency_distribution_math import \
    convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute, \
    convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration


def test_convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute():
    input_log_slope0 = -0.5
    input_log_intercept0 = 2
    input_log_luminosity0 = 0
    output_log_slope0, output_log_intercept0 = (
        convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute(
            log_slope=input_log_slope0, log_intercept=input_log_intercept0, log_luminosity=input_log_luminosity0))
    assert output_log_slope0 == -0.5
    assert output_log_intercept0 == 2
    input_log_slope1 = -0.5
    input_log_intercept1 = 2
    input_log_luminosity1 = 2
    output_log_slope1, output_log_intercept1 = (
        convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute(
            log_slope=input_log_slope1, log_intercept=input_log_intercept1, log_luminosity=input_log_luminosity1))
    assert output_log_slope1 == -0.5
    assert output_log_intercept1 == 3
    input_log_slope2 = -0.5
    input_log_intercept2 = 2
    input_log_luminosity2 = -1
    output_log_slope2, output_log_intercept2 = (
        convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute(
            log_slope=input_log_slope2, log_intercept=input_log_intercept2, log_luminosity=input_log_luminosity2))
    assert output_log_slope2 == -0.5
    assert output_log_intercept2 == 1.5
    input_log_slope3 = -3
    input_log_intercept3 = 2
    input_log_luminosity3 = 1
    output_log_slope3, output_log_intercept3 = (
        convert_log_flare_frequency_distribution_from_equivalent_duration_to_absolute(
            log_slope=input_log_slope3, log_intercept=input_log_intercept3, log_luminosity=input_log_luminosity3))
    assert output_log_slope3 == -3
    assert output_log_intercept3 == 5


def test_convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration():
    input_log_slope0 = -0.5
    input_log_intercept0 = 2
    input_log_luminosity0 = 0
    output_log_slope0, output_log_intercept0 = (
        convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration(
            log_slope=input_log_slope0, log_intercept=input_log_intercept0, log_luminosity=input_log_luminosity0))
    assert output_log_slope0 == -0.5
    assert output_log_intercept0 == 2
    input_log_slope1 = -0.5
    input_log_intercept1 = 3
    input_log_luminosity1 = 2
    output_log_slope1, output_log_intercept1 = (
        convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration(
            log_slope=input_log_slope1, log_intercept=input_log_intercept1, log_luminosity=input_log_luminosity1))
    assert output_log_slope1 == -0.5
    assert output_log_intercept1 == 2
    input_log_slope2 = -0.5
    input_log_intercept2 = 1.5
    input_log_luminosity2 = -1
    output_log_slope2, output_log_intercept2 = (
        convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration(
            log_slope=input_log_slope2, log_intercept=input_log_intercept2, log_luminosity=input_log_luminosity2))
    assert output_log_slope2 == -0.5
    assert output_log_intercept2 == 2
    input_log_slope3 = -3
    input_log_intercept3 = 5
    input_log_luminosity3 = 1
    output_log_slope3, output_log_intercept3 = (
        convert_log_flare_frequency_distribution_from_absolute_to_equivalent_duration(
            log_slope=input_log_slope3, log_intercept=input_log_intercept3, log_luminosity=input_log_luminosity3))
    assert output_log_slope3 == -3
    assert output_log_intercept3 == 2
