import jax.numpy as jnp
import numpy as np
from promap import fit
from promap.parameter_ranges import ParameterRanges
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
from promap.hyper_parameters import HyperParameters
import unittest


class TestFit(unittest.TestCase):
    def test_initial_guess(self):
        f_model = FluorescenceModel(mu_i=100, mu_b=50, sigma=0.03)
        t_model = TraceModel(f_model, p_on=0.05, p_off=0.05)
        trace, states = t_model.generate_trace(1, 10, 100)

        trace = jnp.round(trace)
        print(trace)
        print(f"Maximum value in trace: {trace.max()}")

        hyper_parameters_discrete = HyperParameters(
            y_low=1,
            gradient_step_size=1e-3,
            num_guesses=1,
            epoch_length=1000,
            is_done_limit=1e-5,
            mu_gradient_step_size=1e-3,
            distribution_threshold=1e-1,
            discretize_x=True,
            discrete_x_dtype='uint8',
            max_x_value=255)

        likelihood = fit.get_likelihood(
            y=1,
            trace=trace,
            parameters= jnp.asarray([100, 50, 0.03, 0.05, 0.05]),
            hyper_parameters=hyper_parameters_discrete)
        print(likelihood)

        hyper_parameters = HyperParameters(
            y_low=1,
            gradient_step_size=1e-3,
            num_guesses=1,
            epoch_length=1000,
            is_done_limit=1e-5,
            mu_gradient_step_size=1e-3,
            distribution_threshold=1e-1,
            discretize_x=False)

        likelihood = fit.get_likelihood(
            y=1,
            trace=trace,
            parameters= jnp.asarray([100, 50, 0.03, 0.05, 0.05]),
            hyper_parameters=hyper_parameters)
        print(likelihood)

        assert False
