class HyperParameters:
    """Hyperparameters to control the maximum likelihood optimization.

    Args:

        gradient_step_size (float):

            The step size for the gradient optimization of the parameters.

        num_guesses (int):

            The number of parameter guesses to start maximum likelihood
            optimization from.

        epoch_length (int):

            The length of an "epoch". This is used to break the optimization
            into smaller stretches (epochs). Convergence is tested for at the
            end of each epoch.

        discretize_x (``bool``):

            If to discretize the measurements. Discretization will consume less
            memory, but is not as accurate as using the original measurements.

        discrete_x_dtype (``jnp.dtype``):

            Datatype of the discretized x values (``jax.numpy.uint8`` or
            ``jax.numpy.uint16``).

        max_x_value (``int``):

            The maximal x value to consider for discretizing x.
    """


    def __init__(
            self,
            y_low=1,
            gradient_step_size=1e-3,
            num_guesses=5,
            epoch_length=1000,
            is_done_limit=1e-5,
            mu_gradient_step_size=1e-3,
            distribution_threshold=1e-1,
            discretize_x=False,
            discrete_x_dtype='uint16',
            max_x_value=None):

        self.y_low = y_low
        self.gradient_step_size = gradient_step_size
        self.num_guesses = num_guesses
        self.epoch_length = epoch_length
        self.is_done_limit = is_done_limit
        self.mu_gradient_step_size = mu_gradient_step_size
        self.distribution_threshold=distribution_threshold
        self.discretize_x = discretize_x
        self.discrete_x_dtype = discrete_x_dtype
        self.max_x_value = max_x_value
