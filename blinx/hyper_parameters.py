from .parameters import Parameters


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

            a weight to account for outlier, or out of distribution intensity
            measurements. Occasional measurements contain extreme noise and
            this sets a minimum possible probability

        delta_t (float, defaul=200):

            the exposure time of a single frame in ms
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
        sigma_scale=None
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
        self.delta_t = delta_t
        self.r_e_loc = r_e_loc
        self.r_e_scale = r_e_scale
        self.r_bg_loc = r_bg_loc
        self.r_bg_scale = r_bg_scale
        self.g_loc = g_loc
        self.g_scale = g_scale
        self.mu_loc = mu_loc
        self.mu_scale = mu_scale
        self.sigma_loc = sigma_loc
        self.sigma_scale = sigma_scale

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
