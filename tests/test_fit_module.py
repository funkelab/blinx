import jax.numpy as jnp
import numpy as np
from promap import fit
from promap.parameter_ranges import ParameterRanges
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
from promap.hyper_parameters import HyperParameters
import unittest


class TestFit(unittest.TestCase):
    def test_find_minima_nd(self):
        '''
        Test that find_minima_3d correctly finds the indecies of a single
        local minima of a 3D array
        '''
        test_array = np.ones((100, 1, 10, 10, 1))*2
        minima = jnp.asarray([25, 0, 2, 9, 0])
        test_array[tuple(minima)] = 1
        test_array = jnp.asarray(test_array)
        minima_found = fit._find_minima_nd(test_array, num_minima=1)
        result = minima == minima_found

        self.assertTrue(result.all())

        return

    def test_initial_guess(self):
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on=0.05, p_off=0.05)
        trace, states = t_model.generate_trace(4, 10, 1000)

        parameter_ranges = ParameterRanges(
            mu_range=(1000, 3000),
            mu_bg_range=(5000, 5000),
            sigma_range=(0.02, 0.1),
            p_on_range=(0.02, 0.08),
            p_off_range=(0.02, 0.08),
            mu_step=5,
            mu_bg_step=1,
            sigma_step=9,
            p_on_step=7,
            p_off_step=7)

        initial_guesses = fit.get_initial_guesses(
            y=4,
            trace=trace,
            parameter_ranges=parameter_ranges,
            num_guesses=1)

        true_vals = jnp.asarray([2000, 5000, 0.03, 0.05, 0.05])
        result = jnp.isclose(true_vals, initial_guesses)

        self.assertTrue(result.all())
        return

    def test_find_y(self):
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on=0.05, p_off=0.05)

        parameter_ranges = ParameterRanges(
            mu_range=(1000, 3000),
            mu_bg_range=(5000, 5000),
            sigma_range=(0.1, 0.1),
            p_on_range=(0.001, 0.1),
            p_off_range=(0.001, 0.1),
            mu_step=5,
            mu_bg_step=1,
            sigma_step=1,
            p_on_step=5,
            p_off_step=5)
        hyper_parameters = HyperParameters(
            gradient_step_size=1e-3,
            num_guesses=1,  # all the parameters above
            epoch_length=1000)
        traces = []
        for seed in range(2):
            trace, states = t_model.generate_trace(4, seed, 1000)
            traces.append(trace)
        traces = jnp.array(traces)
        ys, parameters, likelihoods = fit.most_likely_ys(
            traces,
            y_low=2,
            y_high=6,
            parameter_ranges=parameter_ranges,
            hyper_parameters=hyper_parameters)
        result = ys == jnp.asarray([4, 4])

        self.assertTrue(result.all())

        return


if __name__ == '__main__':
    unittest.main()
