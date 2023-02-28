import jax.numpy as jnp
import numpy as np
from promap import fit
from promap.parameter_ranges import ParameterRanges
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
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
        t_model = TraceModel(f_model, p_on=0.1, p_off=0.02)
        trace, states = t_model.generate_trace(4, 11, 1000)

        likelihoods = []
        for y in range(2, 7):
            a, b = fit.optimize_params(
                y,
                trace=trace,
                initial_params=None)
            likelihoods = np.append(likelihoods, a)
            print(f'y:{y}  likelihood: {a:.2f})')

        self.assertTrue(np.argmax(likelihoods) == 2)

        return


if __name__ == '__main__':
    unittest.main()
