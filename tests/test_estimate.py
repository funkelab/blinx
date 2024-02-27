import jax.numpy as jnp
from blinx.parameters import Parameters
from blinx import HyperParameters, ParameterRanges, create_step_sizes
from blinx.estimate import estimate_parameters, get_initial_parameter_guesses


def test_parameter_guesses(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = traces.max()
    hyper_parameters.num_guesses = 1

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges(
        r_e_range=(parameters.r_e - 1.0, parameters.r_e + 1.0),
        r_bg_range=(parameters.r_bg - 1.0, parameters.r_bg + 1.0),
        mu_ro_range=(parameters.mu_ro, parameters.mu_ro),
        sigma_ro_range=(parameters.sigma_ro, parameters.sigma_ro),
        gain_range=(parameters.gain, parameters.gain),
        p_on_range=(parameters.p_on, parameters.p_on),
        p_off_range=(parameters.p_off, parameters.p_off),
        r_e_step=3,
        r_bg_step=3,
        mu_ro_step=1,
        sigma_ro_step=1,
        gain_step=1,
        p_on_step=1,
        p_off_step=1,
    )

    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )

    # should return only one guess
    assert len(parameter_guesses.r_e) == 1

    # ...and that should be the true one
    assert parameter_guesses.r_e == parameters.r_e
    assert parameter_guesses.r_bg == parameters.r_bg


def test_r_e_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    # traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        r_e=1e-3, r_bg=0, mu_ro=0, sigma_ro=0, gain=0, p_on=0, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges()
    initial_parameters = Parameters(
        r_e=jnp.expand_dims(jnp.repeat(4.5, 1), axis=1),
        r_bg=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        mu_ro=jnp.expand_dims(jnp.repeat(5000.0, 1), axis=1),
        sigma_ro=jnp.expand_dims(jnp.repeat(1000.0, 1), axis=1),
        gain=jnp.expand_dims(jnp.repeat(2.0, 1), axis=1),
        p_on=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        p_off=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        probs_are_logits=False,
    )
    fit_parameters, likelihood, evidence = estimate_parameters(
        trace,
        y,
        parameter_ranges,
        hyper_parameters,
        initial_parameters=initial_parameters,
    )

    print(fit_parameters[0].r_e)
    assert jnp.isclose(fit_parameters[0].r_e, parameters.r_e, atol=0.5)


def test_r_bg_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    # traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        r_e=0, r_bg=1e-2, mu_ro=0, sigma_ro=0, gain=0, p_on=0, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges()
    initial_parameters = Parameters(
        r_e=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        r_bg=jnp.expand_dims(jnp.repeat(4.0, 1), axis=1),
        mu_ro=jnp.expand_dims(jnp.repeat(5000.0, 1), axis=1),
        sigma_ro=jnp.expand_dims(jnp.repeat(1000.0, 1), axis=1),
        gain=jnp.expand_dims(jnp.repeat(2.0, 1), axis=1),
        p_on=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        p_off=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        probs_are_logits=False,
    )
    fit_parameters, likelihood, evidence = estimate_parameters(
        trace,
        y,
        parameter_ranges,
        hyper_parameters,
        initial_parameters=initial_parameters,
    )

    print(fit_parameters[0].r_e)
    assert jnp.isclose(fit_parameters[0].r_bg, parameters.r_bg, atol=0.5)


def test_p_on_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    # traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        r_e=0, r_bg=0, mu_ro=0, sigma_ro=0, gain=0, p_on=1e-3, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges()
    initial_parameters = Parameters(
        r_e=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        r_bg=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        mu_ro=jnp.expand_dims(jnp.repeat(5000.0, 1), axis=1),
        sigma_ro=jnp.expand_dims(jnp.repeat(1000.0, 1), axis=1),
        gain=jnp.expand_dims(jnp.repeat(2.0, 1), axis=1),
        p_on=jnp.expand_dims(jnp.repeat(0.1, 1), axis=1),
        p_off=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        probs_are_logits=False,
    )
    fit_parameters, likelihood, evidence = estimate_parameters(
        trace,
        y,
        parameter_ranges,
        hyper_parameters,
        initial_parameters=initial_parameters,
    )

    print(fit_parameters[0].r_e)
    assert jnp.isclose(fit_parameters[0].p_on, parameters.p_on, atol=0.5)


def test_p_off_inference(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    # traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
    hyper_parameters.num_guesses = 1
    hyper_parameters.step_sizes = create_step_sizes(
        r_e=0, r_bg=0, mu_ro=0, sigma_ro=0, gain=0, p_on=1e-3, p_off=0
    )
    hyper_parameters.epoch_length = 1000

    # create parameter ranges that contain the true values
    parameter_ranges = ParameterRanges()
    initial_parameters = Parameters(
        r_e=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        r_bg=jnp.expand_dims(jnp.repeat(5.0, 1), axis=1),
        mu_ro=jnp.expand_dims(jnp.repeat(5000.0, 1), axis=1),
        sigma_ro=jnp.expand_dims(jnp.repeat(1000.0, 1), axis=1),
        gain=jnp.expand_dims(jnp.repeat(2.0, 1), axis=1),
        p_on=jnp.expand_dims(jnp.repeat(0.05, 1), axis=1),
        p_off=jnp.expand_dims(jnp.repeat(0.1, 1), axis=1),
        probs_are_logits=False,
    )
    fit_parameters, likelihood, evidence = estimate_parameters(
        trace,
        y,
        parameter_ranges,
        hyper_parameters,
        initial_parameters=initial_parameters,
    )

    print(fit_parameters[0].r_e)
    assert jnp.isclose(fit_parameters[0].p_off, parameters.p_off, atol=0.5)
