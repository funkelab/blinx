import pytest
from blinx.parameters import Parameters
from blinx.trace_model import generate_trace


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

@pytest.fixture(scope="module")
def trace_with_groundtruth_noisy():
    """Create a random trace for other tests to work on."""

    y = 5
    parameters = Parameters(mu=100.0, mu_bg=10.0, sigma=0.03, p_on=0.05, p_off=0.05)
    num_frames = 1000

    trace, zs = generate_trace(y, parameters, num_frames)

    return {
        "trace": trace,
        "parameters": parameters,
        "y": y,
        "zs": zs,
    }
