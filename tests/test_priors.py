import jax.numpy as jnp
import numpy as np
import pytest
from blinx.hyper_parameters import HyperParameters


def test_priors():
    hps = HyperParameters(
        r_e_loc=None,  # No priors for that parameter
        r_e_scale=None,  # if either loc or scale is None other must be as well
        r_bg_loc=1,  # does it convert an int to the correct shape jax array
        r_bg_scale=1,
        g_loc=[1],  # does it convert a list to the correct shape jax array
        g_scale=[1],
        mu_loc=0.75,  # does it convert a float to the correct shape jax array
        mu_scale=0.1,
        sigma_loc=jnp.array(1),  # if the input is already a jax array
        sigma_scale=1,
        num_traces=5,
    )

    # check that None priors remain None after re-shaping
    np.testing.assert_equal(hps.prior_locs.r_e, None)
    np.testing.assert_equal(hps.prior_scales.r_e, None)

    # check that priors have the correct shape
    np.testing.assert_equal(hps.prior_locs.r_bg.shape, (5,))
    np.testing.assert_equal(hps.prior_scales.r_bg.shape, (5,))
    np.testing.assert_equal(hps.prior_locs.gain.shape, (5,))
    np.testing.assert_equal(hps.prior_scales.gain.shape, (5,))
    np.testing.assert_equal(hps.prior_locs.mu_ro.shape, (5,))
    np.testing.assert_equal(hps.prior_scales.mu_ro.shape, (5,))
    np.testing.assert_equal(hps.prior_locs.sigma_ro.shape, (5,))
    np.testing.assert_equal(hps.prior_scales.sigma_ro.shape, (5,))

    return
