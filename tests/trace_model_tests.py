import numpy as np
import unittest
import pandas as pd
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams


class TestTraceModel(unittest.TestCase):
    def test_generate_trace(self):
        sim_trace_len = 100
        trace_simulator = TraceModel(EmissionParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)

        self.assertAlmostEqual(len(trace), sim_trace_len)

        return

    def test_p_trace_given_y(self):
        sim_trace_len = 1000
        trace_simulator = TraceModel(EmissionParams(), 0.1, sim_trace_len)
        trace_simulator.set_params(0.5, 0.5)
        trace = trace_simulator.generate_trace(1)

        probability = trace_simulator.p_trace_given_y(trace, 1)

        self.assertLessEqual(probability, 0)
        
if __name__ == '__main__':
    unittest.main()