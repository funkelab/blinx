from .parameters import Parameters
import jax.numpy as jnp


def create_step_sizes(*args, **kwargs):
    return Parameters(*args, **kwargs, probs_are_logits=True)


class HyperParameters:
    """Hyperparameters used for maximum likelihood optimization.

    Args:

        min_y (float, default=1):

            The minimum number of emitters expected to be in the system

        num_guesses (int, default=1):

            The number of parameter guesses to start maximum likelihood
            optimization from.

        epoch_length (int, default=1000):

            The length of an "epoch". This is used to break the optimization
            into smaller stretches (epochs). Convergence is tested for at the
            end of each epoch.

        is_done_limit (float, default=1e-5):

            the minimum relative change in log_likelihood between iterations,
            below which a plateau is reached and optimization stopped

        is_done_window (int, default=10):

            The number of previous iterations to consider when determining if
            an optimization plateau is reached

        step_sizes (:class:`Parameters`, defaults=(mu=1.0, mu_bg=1.0, sigma=1e-3, p_on=1e-3, p_off=1e-3)):

            The gradient step size used in sgd optimization, individually
            specified for each parameter in :class:`Parameters`

            Importantly all values must be type: float for gradietns to be
            calculated

        distribution_threshold (float, default=1e-1):

            Used in post_process to compare distribution of optimal states to
            the stady state distribution. Useful in filtering bad fits

        max_x (float):

            The maximum intensity value observed in the trace. Used to
            discritize the trace and calculate individual measurement probabilities

        num_x_bins (int, default=1024):

            number of bins to use when discretizing the trace and calcualting
            individual measurement probabilities

        p_outlier (float, default=0.1):

            a weight to account for outliers, or out of distribution intensity
            measurements. Occasional measurements contain extreme noise and
            this sets a minimum possible probability
        
        num_outliers (int, default=20):

            the number of outlier intensities to assign constant likelihoods, 
            removing them from contributing towards the difference in likelihoods between counts.
            i.e. the 20 frames with the highest intensities will be omited

        delta_t (float, defaul=200):

            the exposure time of a single frame in ms
        
        r_e_loc / r_bg_loc / g_loc / mu_loc /sigma_loc (float, default=None):

            Mean (loc) of the prior distribution on each of the fittable parameters. If None a uniform prior is assumed.
            If either loc or scale is given for a parameter, the other must be given as well.
        
        r_e_scale / r_bg_scale / g_scale / mu_scale /sigma_scale (float, default=None): 

            Variance (scale) of the prior distribution on each of the fittable parameters. If None a uniform prior is assumed.
            If either loc or scale is given for a parameter, the other must be given as well.
        

    """

    def __init__(
        self,
        min_y=1,
        num_guesses=1,
        epoch_length=1000,
        is_done_limit=1e-5,
        is_done_window=10,
        step_sizes=create_step_sizes(
            r_e=1.0, r_bg=1.0, mu_ro=1.0, sigma_ro=1e-3, gain=1.0, p_on=1e-3, p_off=1e-3
        ),
        distribution_threshold=1e-1,
        max_x=None,
        num_x_bins=1024,
        p_outlier=0.1,
        num_outliers=20,
        delta_t=200.0,
        r_e_loc=None,
        r_e_scale=None,
        r_bg_loc=None,
        r_bg_scale=None,
        g_loc=None,
        g_scale=None,
        mu_loc=None,
        mu_scale=None,
        sigma_loc=None,
        sigma_scale=None,
    ):
        self.min_y = min_y
        self.num_guesses = num_guesses
        self.epoch_length = epoch_length
        self.is_done_limit = is_done_limit
        self.is_done_window = is_done_window
        self.step_sizes = step_sizes
        self.distribution_threshold = distribution_threshold
        self.max_x = max_x
        self.num_x_bins = num_x_bins
        self.p_outlier = p_outlier
        self.num_outliers = num_outliers
        self.delta_t = delta_t

        # priors
        self.prior_locs = Parameters(
            r_e=r_e_loc,
            r_bg=r_bg_loc,
            mu_ro=mu_loc,
            sigma_ro=sigma_loc,
            gain=g_loc,
            p_on=None,
            p_off=None,
            probs_are_logits=True,
        )
        self.prior_scales = Parameters(
            r_e=r_e_scale,
            r_bg=r_bg_scale,
            mu_ro=mu_scale,
            gain=g_scale,
            sigma_ro=sigma_scale,
            p_on=None,
            p_off=None,
            probs_are_logits=True,
        )

        if sum([r_e_loc is None, r_e_scale is None]) == 1:
            raise RuntimeError("Both r_e_loc and r_e_scale need to be provided")
        if sum([r_bg_loc is None, r_bg_scale is None]) == 1:
            raise RuntimeError("Both r_bg_loc and r_bg_scale need to be provided")
        if sum([g_loc is None, g_scale is None]) == 1:
            raise RuntimeError("Both g_loc and g_scale need to be provided")
        if sum([mu_loc is None, mu_scale is None]) == 1:
            raise RuntimeError("Both mu_loc and mu_scale need to be provided")
        if sum([sigma_loc is None, sigma_scale is None]) == 1:
            raise RuntimeError("Both sigma_loc and sigma_scale need to be provided")

    # below is experimental
    # ------------------------------------------------
    def check_length(self, val, target_length):
        if val is None:
            return val
        elif len(val) == 1:
            return jnp.repeat(val, target_length)
        elif len(val) > 1 and len(val) != target_length:
            raise RuntimeError("not enough prior values provided")

    def check_prior_shapes(self, target_length):
        self.prior_locs = Parameters(
            r_e=self.check_length(self.prior_locs.r_e, target_length),
            r_bg=self.check_length(self.prior_locs.r_bg, target_length),
            mu_ro=self.check_length(self.prior_locs.mu_ro, target_length),
            sigma_ro=self.check_length(self.prior_locs.sigma_ro, target_length),
            gain=self.check_length(self.prior_locs.gain, target_length),
            p_on=self.check_length(self.prior_locs.p_on, target_length),
            p_off=self.check_length(self.prior_locs.p_off, target_length),
        )
        self.prior_scales = Parameters(
            r_e=self.check_length(self.prior_scales.r_e, target_length),
            r_bg=self.check_length(self.prior_scales.r_bg, target_length),
            mu_ro=self.check_length(self.prior_scales.mu_ro, target_length),
            sigma_ro=self.check_length(self.prior_scales.sigma_ro, target_length),
            gain=self.check_length(self.prior_scales.gain, target_length),
            p_on=self.check_length(self.prior_scales.p_on, target_length),
            p_off=self.check_length(self.prior_scales.p_off, target_length),
        )
