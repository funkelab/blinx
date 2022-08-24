import numpy as np
import unittest
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams

class TestTraceModel(unittest.TestCase):
    def test_prediction_consistancy(self):
        # generate trace
        sim_trace_len = 2000
        trace_simulator = TraceModel(EmissionParams(mu_i=100, mu_b=100)
                                     , 0.1, sim_trace_len)
        trace_simulator.set_params(0.05, 0.1)
        trace = trace_simulator.generate_trace(3)
        
        # grid search 
        e_params = EmissionParams(mu_i=100, sigma_i=0.2, mu_b=100)
        t_model = TraceModel(e_params, 0.1, len(trace))
        ys = [2, 3, 4]

        points = 5
        best_ps = np.zeros((len(ys),2))
        probs = np.zeros((len(ys)))

        for i, y in enumerate(ys):
            t_model.p_on = None
            t_model.p_off = None
            best_ps[i,0], best_ps[i,1] = t_model._line_search_params(trace, y,
                                                                     points=points,
                                                                     p_on_max=0.2,
                                                                     p_off_max=0.2)
            probs[i] = t_model.p_trace_given_y(trace, y)
            
        self.assertAlmostEqual(np.argmax(probs), 1)
        self.assertAlmostEqual(best_ps[1,0], 0.05, places=3)
        self.assertAlmostEqual(best_ps[1,1], 0.1, places=3)



if __name__ == '__main__':
    unittest.main()