import jax.numpy as jnp
import numpy as np
from promap import fit
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
import unittest


class TestFit(unittest.TestCase):
    def test_find_minima_3d(self):
        ''' 
        Test that find_minima_3d correctly finds the indecies of a single 
        local minima of a 3D array
        '''
        test_array = np.ones((100,100,100))*2
        minima = np.random.randint(0,99,3)
        test_array[tuple(minima)] = 1
        found_minima = fit._find_minima_3d(test_array, 3)
                
        result = jnp.unique(minima) == jnp.unique(found_minima)
            
        self.assertTrue(result.all())

        return

    def optimize_params(self):
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on = 0.05, p_off = 0.05)
        trace, states = t_model.generate_trace(4, 10, 1000)
        
        fit._likelihood_func(y=4, p_on=0.05, p_off=0.05, mu=2000, sigma=0.03,
                             trace=trace, mu_b_guess=5000)
        
        fit.optimize_params(4, trace, initial_params=[[0.1], [0.1], [1000.]],
                            mu_b_guess=5000)

        return
    
    def test_initial_guess(self):
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on = 0.05, p_off = 0.05)
        trace, states = t_model.generate_trace(4, 10, 1000)
        
        initial_guesses = fit._initial_guesses(100, 0.2, y=6, trace=trace,
                                               mu_b_guess = 5000)
        self.assertTrue(initial_guesses)
        return
    
    def test_find_y(self):
        f_model = FluorescenceModel(mu_i=2000, mu_b=5000, sigma_i=0.03)
        t_model = TraceModel(f_model, p_on = 0.1, p_off = 0.02)
        trace, states = t_model.generate_trace(4, 11, 1000)
        
        likelihoods = []
        for y in range(2,7):
            a, b = fit.optimize_params(y, trace = trace,
               initial_params=None)
            likelihoods = np.append(likelihoods, a)
            print(f'y:{y}  likelihood: {a:.2f})')
            
        self.assertTrue(np.argmax(likelihoods) == 2)
        
        return
        
if __name__ == '__main__':
    unittest.main()
    
    
    
