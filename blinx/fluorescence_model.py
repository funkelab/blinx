import jax
import jax.numpy as jnp
from jax import random

from .constants import eps


def p_x_given_z(x_left, x_right, z, mu, mu_bg, sigma, hyper_parameters):
    """
    The probability of observing an intensity within the range x_left to x_right,
    given specific parameters.
    Args:

        x_left (float):

            left bound of the intensity bin

        x_right (float):

            right bound of the intensity bin

        z (int):

            the number of "on" emitters

        mu (float):

            the mean intensity of a single emitter

        mu_bg (float):

            mean background intensity

        sigma (float):

            standard deviation of the intensity of a single emitter

        hyper_parameters (:class:`HyperParameters`):

            The hyper-parameters used for the maximum likelihood estimation

    Returns:
        probability (float):

            probability of observing intensity in range x_left to x_right,
            given 'z' on emitters

    """

    p_outlier = hyper_parameters.p_outlier
    num_bins = hyper_parameters.num_x_bins

    mean = mu * z + mu_bg

    return p_outlier / num_bins + (1 - p_outlier) * p_lognorm(
        x_left, x_right, mean, sigma
    )


def sample_x_given_z(z, mu, mu_bg, sigma, key):
    """Randomly sample an intensity from a distribution.

    Possible intensities are lognormally distributed with the mean and std of the
    underlying nomral distribution given by mu and sigma

    Args:

        z (int):

            the number of "on" emitters

        mu (float):

            the mean intensity of a single emitter

        mu_bg (float):

            mean background intensity

        sigma (float):

            standard deviation of the intensity of a single emitter

        key (jax PRNG key):

            key for the jax random number generator

    Returns:

        intensity (float):
    """
    log_mean = jnp.log(mu * z + mu_bg)
    std_samples = random.normal(key, z.shape)

    return jnp.exp(std_samples * sigma + log_mean)


def p_lognorm(x_left, x_right, mean, sigma):
    # implimentation of the lognormal distribution

    log_mean = jnp.log(mean)

    # ensure the following log's behave well
    x_left = jnp.clip(x_left, a_min=eps)
    x_right = jnp.clip(x_right, a_min=eps)

    log_x_left = jnp.log(x_left)
    log_x_right = jnp.log(x_right)

    cdf_left = jax.scipy.stats.norm.cdf(log_x_left, loc=log_mean, scale=sigma)
    cdf_right = jax.scipy.stats.norm.cdf(log_x_right, loc=log_mean, scale=sigma)

    return cdf_right - cdf_left
