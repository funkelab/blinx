import jax
import jax.numpy as jnp
import numpy as np
from blinx import HyperParameters, ParameterRanges
from blinx.trace_model import get_trace_log_likelihood


def test_gradients(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()

    value_and_gradients = jax.value_and_grad(
        lambda p: get_trace_log_likelihood(trace, y, p, hyper_parameters)
    )

    log_likelihood, gradients = value_and_gradients(parameters)
    gradients = np.asarray(
        [
            gradients.r_e,
            gradients.r_bg,
            gradients.mu_ro,
            gradients.sigma_ro,
            gradients.gain,
            gradients.p_on,
            gradients.p_off,
        ]
    )

    assert not np.any(np.isnan(gradients))


def test_trace_log_likelihood(trace_with_groundtruth_noisy):
    trace = trace_with_groundtruth_noisy["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth_noisy["parameters"]
    y = trace_with_groundtruth_noisy["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
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
    a = parameter_ranges.to_parameters()

    log_likelihood_over_parameters = jax.vmap(
        lambda t, p: get_trace_log_likelihood(t, y, p, hyper_parameters),
        in_axes=(None, 0),
    )

    # vmap over traces
    log_likelihoods = jax.vmap(log_likelihood_over_parameters, in_axes=(0, None))(
        traces, a
    )

    # check that there are no NaNs
    assert not jnp.any(jnp.isnan(jnp.asarray(log_likelihoods)))
