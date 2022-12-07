import jax
import jax.numpy as jnp
from jax import random


class FluorescenceModel:
    '''
    - Deals with the intensity measurements
    - The emmission probabilities of the hidden markov model

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

        distribution (``str``):
            The type of distribution to use to model the intensity values of
            one bound fluorophore. Possible choices: 'lognormal' (default) and
            'poisson'.
    '''

    def __init__(self,
                 mu_i=1,
                 sigma_i=0.1,
                 mu_b=1,
                 sigma_b=0.1,
                 label_eff=1,
                 distribution='lognormal'):

        self.mu_i = mu_i
        self.sigma_i = sigma_i
        self.sigma_i2 = sigma_i**2
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.sigma_b2 = sigma_b**2
        self.label_eff = label_eff
        self.distribution = distribution

    def sample_x_z(self, z, key):
        """Draw sample intensity values given a number of bound fluorophores.

        Args:

            z (tensor of ``int``):
                List containing the number of bound fluorophores.

            key:
                JAX random number generator key.

        Returns:

            A trace of intensity values.
        """

        if self.distribution == 'lognormal':
            return self._sample_x_z_lognorm(z, key)
        elif self.distribution == 'poisson':
            return self._sample_x_z_poisson(z, key)
        else:
            raise RuntimeError(
                f"Unknown distribution type {self.distribution}")

    def p_x_given_zs(self, x, max_z):
        """Compute the probability of observing an intensity value for all
        number of bound fluorophores (``z``) from ``0`` to (including)
        ``max_z``.

        Args:

            x (array of type ``float``):
                The observed intensity values.

            max_z (``int``):
                The maximum number of binding sites.

        Returns:

            Array of shape ``(max_z + 1, n)`` where ``n`` is the number of
            elements in ``x``, containing the emission probabilites for ``z =
            0, ..., max_z``.
        """
        zs = jnp.arange(0, max_z + 1)
        x = jnp.expand_dims(x, 0)

        if self.distribution == 'lognormal':
            result = jax.vmap(self._p_x_given_z_lognorm,
                              in_axes=(1, None))(x, zs)
        elif self.distribution == 'poisson':
            result = jax.vmap(self._p_x_given_z_poisson,
                              in_axes=(1, None))(x, zs)
        else:
            raise RuntimeError(
                f"Unknown distribution type {self.distribution}")

        return result.T

    def _sample_x_z_poisson(self, z, seed):
        ''' Samples a Poisson random variable '''
        lam = self.mu_i * z + self.mu_b
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        value = random.poisson(subkey, lam)

        return value

    def _sample_x_z_lognorm(self, z, key):

        mean = jnp.log(self.mu_i * z + self.mu_b)

        std_value = random.normal(key, z.shape)
        value = (std_value * self.sigma_i) + mean

        return jnp.exp(value)

    def _p_x_given_z_poisson(self, trace, z):
        mu = z * self.mu_i + self.mu_b
        prob = jax.scipy.stats.poisson.pmf(trace, mu=mu)

        return prob

    def _p_x_given_z_lognorm(self, x, z):

        mean = jnp.log(self.mu_i * z + self.mu_b)
        value_1 = jnp.log(x)
        value_2 = value_1 + 1/256

        prob_1 = jax.scipy.stats.norm.cdf(value_1, loc=mean,
                                          scale=self.sigma_i)
        prob_2 = jax.scipy.stats.norm.cdf(value_2, loc=mean,
                                          scale=self.sigma_i)

        prob = jnp.abs(prob_1 - prob_2)

        return prob
