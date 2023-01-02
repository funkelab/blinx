import jax.numpy as jnp
import numpy as np
from promap import fit
import unittest

class TestFit(unittest.TestCase):
    def test_find_minima_3d(self):
        ''' 
        Test that find_minima_3d correctly finds the indecies of a single 
        local minima of a 3D array
        '''
        test_array = np.ones((200,200,200))*2
        minima = np.random.randint(10,190,3)
        test_array[tuple(minima)] = 1
        found_minima = fit._find_minima_3d(test_array, 3)
                
        result = jnp.unique(minima) == jnp.unique(found_minima)
            
        self.assertTrue(result.all())

        return

        
if __name__ == '__main__':
    unittest.main()
    
    
    