import numpy as np
import unittest
import pandas as pd
from promap.fluorescence_model import FluorescenceModel, EmissionParams


class TestFluorescenceModel(unittest.TestCase):
    def test_p_x_given_z(self):
        z = 5
        f_model = FluorescenceModel(EmissionParams(mu_i=100, mu_b=200))

        sample = f_model.sample_x_i_given_z_i(z)

        probs = np.zeros((10))
        for i in range(len(probs)):
            probs[i] = f_model.p_x_i_given_z_i(sample, i)
        most_likely = np.argmax(probs)
        
        self.assertEqual(z, most_likely)
        return
    
if __name__ == '__main__':
    unittest.main()