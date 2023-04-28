from .parameters import Parameters
import jax.numpy as jnp


class ParameterRanges:
    '''Min/max and number of values to explore by the optimizer.

    Args:

        mu_range (tuple):

            The min and max value of `mu` (the intensity increment per
            fluorophore).

        mu_bg_range (tuple):

            The min and max value of `mu_bg` (the background intensity).

        p_on_range (tuple):
        p_off_range (tuple):

            The min and max value of `p_on` and `p_off`. Either one can be
            `None`, in which case the default of 1e-4 and 1 will be used for
            the min and max, respectively.

        mu_step (int):
        p_on_step (int):
        p_off_step (int):

            The number of values to explore between the corrosponding min and
            max values.
    '''

    def __init__(
            self,
            mu_range=(100, 30000),
            mu_bg_range=(5000, 5000),
            sigma_range=(0.1, 0.1),
            p_on_range=(1e-4, 1.0),
            p_off_range=(1e-4, 1.0),
            mu_step=100,
            mu_bg_step=1,
            sigma_step=1,
            p_on_step=20,
            p_off_step=20):

        self.mu_range = Range(*mu_range, mu_step)
        self.mu_bg_range = Range(*mu_bg_range, mu_bg_step)
        self.sigma_range = Range(*sigma_range, sigma_step)
        self.p_on_range = Range(*p_on_range, p_on_step)
        self.p_off_range = Range(*p_off_range, p_off_step)

        if self.p_on_range.start is None:
            self.p_on_range.start = 1e-4
        if self.p_off_range.start is None:
            self.p_off_range.start = 1e-4
        if self.p_on_range.stop is None:
            self.p_on_range.stop = 1.0
        if self.p_off_range.stop is None:
            self.p_off_range.stop = 1.0

    def num_values(self):

        return tuple(
            r.step
            for r in [
                self.mu_range,
                self.mu_bg_range,
                self.sigma_range,
                self.p_on_range,
                self.p_off_range
            ]
        )

    def to_parameters(self):

        range_tensors = {
            "mu": self.mu_range.to_tensor(),
            "mu_bg": self.mu_bg_range.to_tensor(),
            "sigma": self.sigma_range.to_tensor(),
            "p_on": self.p_on_range.to_tensor(),
            "p_off": self.p_off_range.to_tensor()
        }

        values = {
            name: v.flatten()
            for name, v in zip(
                range_tensors.keys(),
                jnp.meshgrid(*range_tensors.values(), indexing='ij')
            )
        }

        return Parameters(**values)


class Range:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def to_tensor(self):
        return jnp.linspace(self.start, self.stop, self.step)
