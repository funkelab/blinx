import jax.numpy as jnp
from blinx import HyperParameters
from blinx.trace_model import log_p_x_parameters


def test_forward(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = trace.max()
    hyper_parameters.num_guesses = 1

    log_likelihood = log_p_x_parameters(trace, y, parameters, hyper_parameters, hyper_parameters.prior_locs, hyper_parameters.prior_scales)
    print(log_likelihood)
    assert jnp.isclose(log_likelihood, -4328.961, atol=0.001)
