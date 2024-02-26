import numpy as np
from blinx import ParameterRanges


def test_to_parameters():
    parameter_ranges = ParameterRanges(
        r_e_range=(1, 4),
        r_bg_range=(5, 5),
        mu_ro_range=(4000,5000),
        sigma_ro_range=(500, 600),
        gain_range=(1.0,2.0),
        p_on_range=(0.0, 1.0),
        p_off_range=(0.0, 1.0),
        r_e_step=4,
        r_bg_step=1,
        mu_ro_step=3,
        sigma_ro_step=3,
        gain_step=2,
        p_on_step=3,
        p_off_step=3,
    )

    parameters = parameter_ranges.to_parameters()

    assert parameters.r_e.shape == (648,)
    assert parameters.r_bg.shape == (648,)
    assert parameters.mu_ro.shape == (648,)
    assert parameters.sigma_ro.shape == (648,)
    assert parameters.gain.shape == (648,)
    assert parameters.p_on.shape == (648,)
    assert parameters.p_off.shape == (648,)

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
            parameters.r_e[100],
            parameters.r_bg[100],
            parameters.mu_ro[100],
            parameters.sigma_ro[100],
            parameters.gain[100],
            parameters.p_on[100],
            parameters.p_off[100],
        ],
        [1, 5, 4500, 600, 2, 0.0, 0.5],
        rtol=1e-5,
    )
