import numpy as np
import unittest
import pandas as pd
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams


class TestTraceModel(unittest.TestCase):
    def test_generate_trace(self):
        # does the mdoel sucessfully generate a trace of the expercted length
        
        sim_trace_len = 1000
        seed = 10
        trace_simulator = TraceModel(EmissionParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.1, 0.1)
        trace, states = trace_simulator.generate_trace(1, seed)

        self.assertAlmostEqual(len(trace), sim_trace_len)

        return

        
if __name__ == '__main__':
    unittest.main()