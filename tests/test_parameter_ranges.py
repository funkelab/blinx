from promap import ParameterRanges
from promap.fit import initial_guesses
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
import jax.numpy as jnp
import numpy as np
import unittest


class TestParameterRanges(unittest.TestCase):

    def test_to_tensor(self):

        parameter_ranges = ParameterRanges(
            mu_range=(1, 4),
            mu_bg_range=(5, 5),
            sigma_range=(0.0, 0.1),
            p_on_range=(0.0, 1.0),
            p_off_range=(0.0, 1.0),
            mu_step=4,
            mu_bg_step=1,
            sigma_step=2,
            p_on_step=10,
            p_off_step=10)

        parameters = parameter_ranges.to_tensor()

        assert parameters.shape == (800, 5)

        # number of configurations:
        #
        #               100th configuration
        #
        # mu    : 4     1
        # mu_bg : 1     5
        # sigma : 2     0.1
        # p_on  : 10    0.0
        # p_off : 20    0.0

        print(parameters[:100])

        np.testing.assert_almost_equal(
            parameters[100],
            [1, 5, 0.1, 0.0, 0.0],
            decimal=4)

    def test_find_minima(self):

        y = 4
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on=0.05, p_off=0.05)
        trace, states = t_model.generate_trace(y, 10, 1000)

        parameter_ranges = ParameterRanges(
            mu_range=(1000, 3000),
            mu_bg_range=(5000, 5000),
            sigma_range=(0.03, 0.03),
            p_on_range=(0.04, 0.06),
            p_off_range=(0.04, 0.06),
            mu_step=5,
            mu_bg_step=1,
            sigma_step=1,
            p_on_step=3,
            p_off_step=3)

        parameters = initial_guesses(y, trace, parameter_ranges)

        # this should return one single local minimum with the correct values
        np.testing.assert_almost_equal(
            parameters,
            [[2000, 5000, 0.03, 0.05, 0.05]])
