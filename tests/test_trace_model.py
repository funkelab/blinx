import jax
import numpy as np
from blinx import HyperParameters
from blinx.trace_model import get_trace_log_likelihood


def test_gradients(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]

    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu

    value_and_gradients = jax.value_and_grad(
        lambda p: get_trace_log_likelihood(trace, y, p, hyper_parameters)
    )

    log_likelihood, gradients = value_and_gradients(parameters)
    gradients = np.asarray(gradients)

    assert not np.any(np.isnan(gradients))
