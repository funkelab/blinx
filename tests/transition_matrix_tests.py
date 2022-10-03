import numpy as np
import unittest
import pandas as pd
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
import jax.numpy as jnp


class TestTransitionMatrix(unittest.TestCase):
    def test_transition_matrix(self):
        y = 5
        e_params = EmissionParams(mu_i = 50, sigma_i = 0.03, mu_b=200, sigma_b=0.15)
        t_model = TraceModel(e_params, 0.1, 1000)
        t_model.set_params(0.02, 0.05)
        
        transition_matrix = t_model.create_transition_matrix(y,
                                                             t_model.p_on,
                                                             t_model.p_off)
        
        # check no zeros in transition_matrix
        
        # check all rows sum to 1
        row_sums = jnp.sum(transition_matrix, axis=1)
        ones = jnp.ones((y+1))
        self.assertEqual(row_sums.all(), ones.all())


if __name__ == '__main__':
    unittest.main()