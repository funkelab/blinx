import jax.numpy as jnp
from blinx import HyperParameters, ParameterRanges
from blinx.estimate import estimate_parameters, get_initial_parameter_guesses


def test_parameter_guesses(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu - 100.0, parameters.mu + 100.0),
        mu_bg_range=(parameters.mu_bg - 5.0, parameters.mu_bg + 5.0),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=3,
        mu_bg_step=3,
        sigma_step=1,
        p_on_step=1,
        p_off_step=1,
    )

    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )

    # should return only one guess
    assert len(parameter_guesses.mu) == 1

    # ...and that should be the true one
    assert parameter_guesses.mu == parameters.mu


def test_inference(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu - 100.0, parameters.mu + 100.0),
        mu_bg_range=(parameters.mu_bg - 5.0, parameters.mu_bg + 5.0),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=3,
        mu_bg_step=3,
        sigma_step=1,
        p_on_step=1,
        p_off_step=1,
    )

    parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )
