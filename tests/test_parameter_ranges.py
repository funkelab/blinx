from promap import ParameterRanges
import numpy as np


def test_to_parameters():
    parameter_ranges = ParameterRanges(
        mu_range=(1, 4),
        mu_bg_range=(5, 5),
        sigma_range=(0.0, 0.1),
        p_on_range=(0.0, 1.0),
        p_off_range=(0.0, 1.0),
        mu_step=4,
        mu_bg_step=1,
        sigma_step=2,
        p_on_step=10,
        p_off_step=10,
    )

    parameters = parameter_ranges.to_parameters()

    for p in parameters:
        assert p.shape == (800,)

    # number of configurations:
    #
    #               100th configuration
    #
    # mu    : 4     1
    # mu_bg : 1     5
    # sigma : 2     0.1
    # p_on  : 10    0.0
    # p_off : 20    0.0

    np.testing.assert_allclose(
        [
            parameters.mu[100],
            parameters.mu_bg[100],
            parameters.sigma[100],
            parameters.p_on[100],
            parameters.p_off[100],
        ],
        [1, 5, 0.1, 0.0, 0.0],
        rtol=1e-5,
    )
