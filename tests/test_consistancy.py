import jax.numpy as jnp
import jax
from blinx import HyperParameters, ParameterRanges
from blinx.estimate import estimate_parameters, get_initial_parameter_guesses
from blinx.parameters import Parameters
from blinx.trace_model import get_trace_log_likelihood, generate_trace


def test_forward(trace_with_groundtruth):
    trace = trace_with_groundtruth["trace"]
    traces = jnp.expand_dims(trace, axis=0)
    parameters = trace_with_groundtruth["parameters"]
    y = trace_with_groundtruth["y"]
    
    hyper_parameters = HyperParameters()
    hyper_parameters.max_x = 2 * y * parameters.mu
    hyper_parameters.num_guesses = 1
   
    log_likelihood = get_trace_log_likelihood(
        trace,
        y,
        parameters,
        hyper_parameters)
    
    assert jnp.isclose(log_likelihood, -1446.202, atol=0.001)
    
