import numpy as np
import math
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

    def sample_x_i_given_z_i(self, z):
        '''
        - simulate the intensity value given a hidden state "z"
        - random sampling is done in log-space
            - sample from a normal distribution
            - exp() at end so final distribution is lognormal

        Args:
            z:
                The number of active/on fluorophores
        '''

        if z == 0:
            signal = -np.inf  # has no contribution once out of log space
        else:
            mean_i = np.log(z * self.mu_i * np.exp(self.sigma_i2 / 2))
            signal = np.random.normal(mean_i, self.sigma_i2)

        mean_b = np.log(self.mu_b)
        background = np.random.normal(mean_b, self.sigma_b2)

        # split into sperate exps because changing background should
        # not change the estimate of mu_i
        return self._bring_out(signal) + self._bring_out(background)
    
    def sample_x_z_poisson_jax(self, z, seed):
        ''' Samples a Poisson random variable '''
        lam = self.mu_i * z +self.mu_b
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        value = random.poisson(subkey, lam)
        
        return value
    
    def sample_x_z_lognorm_jax(self, z, key, shape=(1,1)):
        #key = random.PRNGKey(seed)
        #key, subkey = random.split(key)
        
        mean = jnp.log(self.mu_i * z + self.mu_b)
        
        std_value = random.normal(key, shape)
        value = (std_value * self.sigma_i) + mean
        
        return jnp.exp(value)
    
    def vmap_p_x_given_z_lognorm(self, x, y):
        zs = jnp.arange(0, y+1)
        x = jnp.expand_dims(x, 0)

        result = jax.vmap(self._p_trace_given_z_lognorm, in_axes=(1, None))(x, zs)

        return np.asarray(result).T

    def vmap_p_x_given_z(self, x, y):
        '''
        - returns a probability matrix of
        '''
        zs = jnp.arange(0, y+1)
        x = jnp.expand_dims(x, 0)

        result = jax.vmap(self._p_trace_given_z_i, in_axes=(1, None))(x, zs)

        return np.asarray(result).T

    def p_x_i_given_z_i(self, x_i, z):
        '''
        - calculate the probability that an intensity x_i arrose from hidden
            state z

        Args:
            x_i:
                the intensity value measured at time i
            z_i:
                the number of active/on fluorophores at time i
        '''

        x = self._bring_in(x_i)

        if z == 0:
            mean_i = -np.inf
        else:
            mean_i = np.log(z * self.mu_i * np.exp(self.sigma_i2 / 2))

        mean_b = np.log(self.mu_b)

        mean = np.log(np.exp(mean_i) + np.exp(mean_b))
        sigma2 = self.sigma_i2 + self.sigma_b2

        result = self._integrate_from_cdf(x, mean, sigma2)

        return result

    def _jax_normal_cdf(self, x, mu, sigma2):
        # CDF of the normal function

        return 0.5 * (1 + jax.lax.erf((x - mu)/jnp.sqrt(2 * sigma2)))

    def _jax_integrate_from_cdf(self, x, mu, sigma2):
        # Aproximates the integral of the normal distribution from x:x+1/256

        a = self._jax_normal_cdf(x, mu, sigma2)
        b = self._jax_normal_cdf(x + (1/256), mu, sigma2)
        prob = jnp.abs(a - b)

        return prob

    def _p_trace_given_z_i(self, trace, z):
        x = jnp.log(trace)

        mean_i = jnp.log(z * self.mu_i * jnp.exp(self.sigma_i2 / 2))
        mean_b = jnp.log(self.mu_b)
        mean = jnp.log(np.exp(mean_i) + jnp.exp(mean_b))
        sigma2 = self.sigma_i2 + self.sigma_b2

        result = self._jax_integrate_from_cdf(x, mean, sigma2)

        return result
    
    def _p_trace_given_z_poisson(self, trace, z):
        mu = z * self.mu_i + self.mu_b
        prob = jax.scipy.stats.poisson.pmf(trace, mu=mu)
        
        return prob
    
    def _p_trace_given_z_lognorm(self, x, z):
        
        mean = jnp.log(self.mu_i * z + self.mu_b)
        value_1 = jnp.log(x)
        value_2 = value_1 + 1/256
        
        prob_1 = jax.scipy.stats.norm.cdf(value_1, loc=mean, scale=self.sigma_i)
        prob_2 = jax.scipy.stats.norm.cdf(value_2, loc=mean, scale=self.sigma_i)
        
        prob = jnp.abs(prob_1 - prob_2)
        
        return prob

    def _normal_cdf(self, x, mu, sigma2):
        # CDF of the normal function

        return 0.5 * (1 + math.erf((x - mu)/np.sqrt(2 * sigma2)))

    def _integrate_from_cdf(self, x, mu, sigma2):
        # Aproximates the integral of the normal distribution x to x + 1/256

        a = self._normal_cdf(x, mu, sigma2)
        b = self._normal_cdf(x + (1/256), mu, sigma2)
        prob = np.abs(a - b)

        return prob

    def _bring_in(self, x):

        return np.log(x)

    def _bring_out(self, x):
        return np.exp(x)

    def _normal(self, x, mu, sigma2):
        # PDF of the normal distribution

        return 1.0 / (np.sqrt(2.0 * np.pi * sigma2)) * \
                np.exp(-(x - mu)**2/(2.0 * sigma2))
