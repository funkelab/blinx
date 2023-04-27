from .constants import eps
from jax import random
import jax
import jax.numpy as jnp


def create_emission_distribution(y, mu, mu_bg, sigma, hyper_parameters):

    max_x = hyper_parameters.max_x
    num_bins = hyper_parameters.num_x_bins
    bin_width = max_x / num_bins

    intensities_left = (
        jnp.arange(0, num_bins) * bin_width
    )
    intensities_right = intensities_left + bin_width

    zs = jnp.arange(0, y + 1)

    p_emission_lookup = jax.vmap(
        p_x_given_z,
        in_axes=(0, 0, None, None, None, None, None))(
            intensities_left,
            intensities_right,
            zs,
            mu, mu_bg, sigma,
            hyper_parameters)

    return p_emission_lookup


def discretize_trace(trace, hyper_parameters):
    """Convert a float-valued trace into a trace of bin indices, according
    to the x discretization."""

    num_bins = hyper_parameters.num_x_bins
    max_x = hyper_parameters.max_x
    bin_width = max_x / num_bins

    if num_bins <= 256:
        x_dtype = 'uint8'
    elif num_bins <= 65536:
        x_dtype = 'uint16'
    else:
        x_dtype = 'uint32'

    return (trace / bin_width).astype(x_dtype)


def p_x_given_z(x_left, x_right, z, mu, mu_bg, sigma, hyper_parameters):

    p_outlier = hyper_parameters.p_outlier
    num_bins = hyper_parameters.num_x_bins

    mean = mu * z + mu_bg

    return (
        p_outlier / num_bins +
        (1 - p_outlier) * p_lognorm(x_left, x_right, mean, sigma)
    )


def sample_x_given_z(z, mu, mu_bg, sigma, key):

    log_mean = jnp.log(mu * z + mu_bg)
    std_samples = random.normal(key, z.shape)

    return jnp.exp(std_samples * sigma + log_mean)


def p_lognorm(x_left, x_right, mean, sigma):

    log_mean = jnp.log(mean)

    # ensure the following log's behave well
    x_left = jnp.clip(x_left, a_min=eps)
    x_right = jnp.clip(x_right, a_min=eps)

    log_x_left = jnp.log(x_left)
    log_x_right = jnp.log(x_right)

    cdf_left = jax.scipy.stats.norm.cdf(
        log_x_left,
        loc=log_mean,
        scale=sigma)
    cdf_right = jax.scipy.stats.norm.cdf(
        log_x_right,
        loc=log_mean,
        scale=sigma)

    return cdf_right - cdf_left
