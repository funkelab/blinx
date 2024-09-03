import pytest
from blinx.parameters import Parameters
from blinx import HyperParameters
from blinx.trace_model import generate_trace


@pytest.fixture(scope="module")
def trace_with_groundtruth():
    """Create a random trace for other tests to work on."""

    y = 5
    parameters = Parameters(
        r_e=5.0,
        r_bg=5.0,
        mu_ro=5000.0,
        sigma_ro=1.0,
        gain=2.0,
        p_on=0.1,
        p_off=0.3,
        _probs_are_logits=False,
    )
    hyper_parameters = HyperParameters(max_x=20000)
    num_frames = 1000

    trace, zs = generate_trace(y, parameters, num_frames, hyper_parameters, 4)

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
    parameters = Parameters(
        r_e=5.0,
        r_bg=5.0,
        mu_ro=5000.0,
        sigma_ro=1000.0,
        gain=2.0,
        p_on=0.05,
        p_off=0.05,
        _probs_are_logits=False,
    )
    hyper_parameters = HyperParameters(max_x=20000)
    num_frames = 1000

    trace, zs = generate_trace(y, parameters, num_frames, hyper_parameters, 4)

    return {
        "trace": trace,
        "parameters": parameters,
        "y": y,
        "zs": zs,
    }
