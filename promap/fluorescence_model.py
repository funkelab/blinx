import jax
import jax.numpy as jnp
from jax import random


class EmissionParams:
    '''
    - Stores all parameters needed for the fluorescence model
    - Used to calcualte emission probabilities of HMM

    Args:
        mu_i:
            the mean intensity of a bound fluorophore

        sigma_i:
            the standard deviation of the intensity of a bound fluorophore

        mu_b:
            the mean intensity of the background

        sigma_b:
            the standard deviation of the intensity of the background

        label_eff:
            the labeling efficiency of y
    '''

    def __init__(self,
                 mu_i=1,
                 sigma_i=0.1,
                 mu_b=1,
                 sigma_b=0.1,
                 label_eff=1):

        self.mu_i = mu_i
        self.sigma_i = sigma_i
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.label_eff = label_eff


class FluorescenceModel:
    '''
    - Deals with the intensity measurements
    - The emmission probabilities of the hidden markov model

    Args:
        Emission_Params:
            Instance of class EmissionParams

    '''

    def __init__(self, emission_params):

        self.mu_i = emission_params.mu_i
        self.sigma_i = emission_params.sigma_i
        self.sigma_i2 = emission_params.sigma_i**2
        self.mu_b = emission_params.mu_b
        self.sigma_b = emission_params.sigma_b
        self.sigma_b2 = emission_params.sigma_b**2
        self.label_eff = emission_params.label_eff

    def sample_x_z_poisson_jax(self, z, seed):
        ''' Samples a Poisson random variable '''
        lam = self.mu_i * z + self.mu_b
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        value = random.poisson(subkey, lam)

        return value

    def sample_x_z_lognorm_jax(self, z, key, shape=(1, 1)):

        mean = jnp.log(self.mu_i * z + self.mu_b)

        std_value = random.normal(key, shape)
        value = (std_value * self.sigma_i) + mean

        return jnp.exp(value)

    def vmap_p_x_given_z_lognorm(self, x, y):
        zs = jnp.arange(0, y+1)
        x = jnp.expand_dims(x, 0)

        result = jax.vmap(self._p_trace_given_z_lognorm,
                          in_axes=(1, None))(x, zs)

        return result.T

    def vmap_p_x_given_z_poisson(self, x, y):
        zs = jnp.arange(0, y+1)
        x = jnp.expand_dims(x, 0)

        result = jax.vmap(self._p_trace_given_z_poisson,
                          in_axes=(1, None))(x, zs)

        return result.T

    def _p_trace_given_z_poisson(self, trace, z):
        mu = z * self.mu_i + self.mu_b
        prob = jax.scipy.stats.poisson.pmf(trace, mu=mu)

        return prob

    def _p_trace_given_z_lognorm(self, x, z):

        mean = jnp.log(self.mu_i * z + self.mu_b)
        value_1 = jnp.log(x)
        value_2 = value_1 + 1/256

        prob_1 = jax.scipy.stats.norm.cdf(value_1, loc=mean,
                                          scale=self.sigma_i)
        prob_2 = jax.scipy.stats.norm.cdf(value_2, loc=mean,
                                          scale=self.sigma_i)

        prob = jnp.abs(prob_1 - prob_2)

        return prob
