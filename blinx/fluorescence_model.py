import jax
import jax.numpy as jnp
from jax import random

from .constants import eps

def p_x_given_z(
    x_left, x_right, z, r_e, r_bg, mean_ro, sigma_ro, gain, hyper_parameters
):
    """
    The probability of observing an intensity within the range x_left to x_right,
    given specific parameters.
    Args:

        x_left (float):

            left bound of the intensity bin (ADU)

        x_right (float):

            right bound of the intensity bin (ADU)

        z (int):

            the number of "on" emitters

        r_e (float):

            the photon emission rate of an emitter (e-/ms)

        r_bg (float):

            the photon emission rate of the background (e-/ms)

        mean_ro (float):

            the mean readout noise of the spot, also known as the offset value (ADU)

        sigma_ro (float):

            the variance of the readout noise of the spot

        gain (float):

            the amplification / conversion factor between ADU and photoelectrons (ADU/e-)

        hyper_parameters (:class:`HyperParameters`):

            The hyper-parameters used for the maximum likelihood estimation

    Returns:
        probability (float):

            probability of observing intensity in range x_left to x_right,
            given 'z' on emitters

    """
    p_outlier = hyper_parameters.p_outlier
    num_bins = hyper_parameters.num_x_bins

    delta_t = hyper_parameters.delta_t

    x_tilda_left = (x_left - mean_ro) / gain + sigma_ro / gain**2

    x_tilda_right = (x_right - mean_ro) / gain + sigma_ro / gain**2

    loc = (z * r_e + r_bg) * delta_t + sigma_ro / gain**2
    scale = jnp.sqrt((z * r_e + r_bg) * delta_t + sigma_ro / gain**2)

    return p_outlier / num_bins + (1 - p_outlier) * p_norm(
        x_tilda_left, x_tilda_right, loc, scale
    )


def p_norm(x_tilda_left, x_tilda_right, loc, scale):
    # implimnetation of the normal distribution

    cdf_left = jax.scipy.stats.norm.cdf(x_tilda_left, loc=loc, scale=scale)
    cdf_right = jax.scipy.stats.norm.cdf(x_tilda_right, loc=loc, scale=scale)

    return cdf_right - cdf_left


def sample_x_given_z(z, r_e, r_bg, mean_ro, sigma_ro, gain, key, hyper_parameters):
    """
    Randomly sample intensity values from a normal distribution.

    Args:

        z (int):

            the number of "on" emitters

        r_e (float):

            the photon emission rate of an emitter (e-/ms)

        r_bg (float):

            the photon emission rate of the background (e-/ms)

        mean_ro (float):

            the mean readout noise of the spot, also known as the offset value (ADU)

        sigma_ro (float):

            the variance of the readout noise of the spot

        gain (float):

            the amplification / conversion factor between ADU and photoelectrons (ADU/e-)

        key (jax PRNG key):

            key for the jax random number generator

        hyper_parameters (:class:`HyperParameters`):

                The hyper-parameters used for the maximum likelihood estimation

    Returns:

        x (float):

            The observed intenstiy (ADU)
    """
    std_samples = random.normal(key, z.shape)

    delta_t = hyper_parameters.delta_t

    loc = (z * r_e + r_bg) * delta_t + sigma_ro / gain**2
    scale = jnp.sqrt((z * r_e + r_bg) * delta_t + sigma_ro / gain**2)

    x = ((std_samples * scale + loc) - sigma_ro / gain**2) * gain + mean_ro

    return x
