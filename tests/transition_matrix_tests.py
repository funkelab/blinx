import numpy as np
import unittest
import pandas as pd
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap.transition_matrix import TransitionMatrix
import jax.numpy as jnp
from scipy import stats


def create_transition_m_truth(y, p_on, p_off):

    size = y+1  # possible states range from 0 - y inclusive
    transition_m = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = 0
            for z in range(i+1):
                p += stats.binom.pmf(z, i, p_off) * \
                    stats.binom.pmf(j-i+z, y-i, p_on)
            transition_m[i, j] = p

    return transition_m 


class TestTransitionMatrix(unittest.TestCase):

    def test_transition_matrix(self):
        y = 5
        p_on = 0.1
        p_off = 0.1
        
        TransMatrix = TransitionMatrix()
        transition_matrix = TransMatrix.create_transition_matrix(y,
                                                                 p_on,
                                                                 p_off)
        
        # check no zeros in transition_matrix
        self.assertTrue(transition_matrix.all())
        
        # check all rows sum to 1
        row_sums = jnp.sum(transition_matrix, axis=1)
        ones = jnp.ones((y+1))
        self.assertEqual(row_sums.all(), ones.all())

    def test_transition_probs(self):
        y = 5
        p_on = 0.1
        p_off = 0.1
        
        TransMatrix = TransitionMatrix()
        transition_matrix_jax = TransMatrix.create_transition_matrix(y,
                                                                     p_on,
                                                                     p_off)
        
        transition_matrix_truth = jnp.asarray(create_transition_m_truth(y,
                                                            p_on,
                                                            p_off))
        
        arrays_match = jnp.unique(jnp.isclose(transition_matrix_jax, 
                                       transition_matrix_truth))
        self.assertTrue(arrays_match)
    

if __name__ == '__main__':
    unittest.main()