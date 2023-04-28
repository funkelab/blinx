from promap.parameters import Parameters
from promap.trace_model import generate_trace
import pytest


@pytest.fixture(scope="module")
def trace_with_groundtruth():
    """Create a random trace for other tests to work on."""

    y = 5
    parameters = Parameters(mu=100.0, mu_bg=10.0, sigma=0.001, p_on=0.1, p_off=0.3)
    num_frames = 1000

    trace, zs = generate_trace(y, parameters, num_frames)

    return {
        "trace": trace,
        "parameters": parameters,
        "y": y,
        "zs": zs,
    }
