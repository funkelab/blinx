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
    """


    def __init__(
            self,
            gradient_step_size=1e-3,
            num_guesses=5,
            epoch_length=1000):

        self.gradient_step_size = gradient_step_size
        self.num_guesses = num_guesses
        self.epoch_length = epoch_length
