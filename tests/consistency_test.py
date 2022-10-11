import numpy as np
import unittest
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap import fit

class TestTraceModel(unittest.TestCase):
    def test_prediction_consistancy(self):
        # generate a test trace
        y_test = 5
        seed = 100
        e_params = EmissionParams(mu_i=50, sigma_i=0.03, mu_b=200, sigma_b=0.15)
        t_model_t = TraceModel(e_params, p_on=0.05, p_off=0.05)
        x_trace, states = t_model_t.generate_trace(y_test, seed=seed,
                                                   num_frames=4000)

        # Calc likelyhood that trace arrose from different y values
        y = 5

        likelihood, p_on, p_off, mu, sigma = fit.optimize_params(y, x_trace)

        print('- '*20)
        print(f'y = {y}')
        print(f'log likelihood   = { likelihood:.2f}')
        print(f'p_on / p_off     = { p_on:.4f} / {p_off:.4f}')
        print(f'mu / sigma       = {mu:.4f} / {sigma:.4f}')


        self.assertAlmostEqual(likelihood, -16592.64, places=2)

if __name__ == '__main__':
    unittest.main()