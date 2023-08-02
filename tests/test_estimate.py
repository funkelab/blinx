import jax.numpy as jnp
from blinx import HyperParameters, ParameterRanges, create_step_sizes
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
    assert parameter_guesses.mu_bg == parameters.mu_bg


def test_parameter_guesses_mismatch(trace_with_groundtruth_noisy):
    # does not pass this test yet...
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu - 90, parameters.mu + 90),
        mu_bg_range=(parameters.mu_bg, parameters.mu_bg),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=20,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=5,
        p_off_step=5,
    )

    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )

    test_parameters = parameter_ranges.to_parameters()
    correct_index = jnp.argmin(jnp.abs(test_parameters.mu - parameters.mu))

    assert test_parameters.mu[correct_index] == parameter_guesses.mu


def test_mu_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        mu=1e-4, mu_bg=0, sigma=0, p_on=0, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu + 10.0, parameters.mu + 11.0),
        mu_bg_range=(parameters.mu_bg, parameters.mu_bg),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=2,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=1,
        p_off_step=1,
    )

    fit_parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )

    assert jnp.isclose(fit_parameters[0].mu, parameters.mu, atol=0.5)


def test_mu_bg_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        mu=0, mu_bg=1e-4, sigma=0, p_on=0, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu, parameters.mu + 1),
        mu_bg_range=(parameters.mu_bg + 10, parameters.mu_bg + 11),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=2,
        mu_bg_step=2,
        sigma_step=1,
        p_on_step=1,
        p_off_step=1,
    )

    fit_parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )

    assert jnp.isclose(fit_parameters[0].mu_bg, parameters.mu_bg, atol=0.5)


def test_sigma_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        mu=0, mu_bg=0, sigma=1e-8, p_on=0, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu, parameters.mu + 1),
        mu_bg_range=(parameters.mu_bg, parameters.mu_bg),
        sigma_range=(parameters.sigma + 0.05, parameters.sigma + 0.06),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=2,
        mu_bg_step=1,
        sigma_step=2,
        p_on_step=1,
        p_off_step=1,
    )

    fit_parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )

    assert jnp.isclose(fit_parameters[0].sigma, parameters.sigma, atol=0.01)


def test_p_on_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        mu=0, mu_bg=0, sigma=0, p_on=1e-3, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu, parameters.mu + 1),
        mu_bg_range=(parameters.mu_bg + 10, parameters.mu_bg + 11),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on + 0.05, parameters.p_on + 0.06),
        p_off_range=(parameters.p_off, parameters.p_off),
        mu_step=2,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=2,
        p_off_step=1,
    )

    fit_parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )

    assert jnp.isclose(fit_parameters[0].p_on, parameters.p_on, atol=0.01)


def test_p_off_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        mu=0, mu_bg=0, sigma=0, p_on=0, p_off=1e-3
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        mu_range=(parameters.mu, parameters.mu + 1),
        mu_bg_range=(parameters.mu_bg + 10, parameters.mu_bg + 11),
        sigma_range=(parameters.sigma, parameters.sigma),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off + 0.05, parameters.p_off + 0.06),
        mu_step=2,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=1,
        p_off_step=2,
    )

    fit_parameters, likelihood = estimate_parameters(
        traces, y, parameter_ranges, hyper_parameters
    )

    assert jnp.isclose(fit_parameters[0].p_off, parameters.p_off, atol=0.01)
