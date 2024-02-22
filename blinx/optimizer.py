from collections import namedtuple

import jax
from optax._src.transform import scale_by_adam

Optimizer = namedtuple("Optimizer", ["init", "step"])


def create_optimizer(value_grad_func, hyper_parameters):
    """A simple gradient ascent optimizer."""

    step_sizes = hyper_parameters.step_sizes

    def init(parameters):
        pass

    def step(trace, parameters, opt_state):
        # get value and gradient

        value, gradients = value_grad_func(trace, parameters)
        # update parameters
        parameters = jax.tree_util.tree_map(
            lambda p, s, g: p + s * g, parameters, step_sizes, gradients
        )

        # return updated parameters, current value, and optimizer state
        return parameters, value, opt_state, gradients

    return Optimizer(init, step)


def create_adam_optimizer(
    value_grad_func,
    hyper_parameters,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    eps_root=0.0,
    mu_dtype=None,
):
    """The Adam optimizer for maximization of the given function."""

    step_sizes = hyper_parameters.step_sizes

    adam_transform = scale_by_adam(
        b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype
    )

    def init(parameters):
        return adam_transform.init(parameters)

    def step(trace, parameters, opt_state, p_loc, p_scale):
        # get value and gradient
        value, gradients = value_grad_func(trace, parameters, p_loc, p_scale)
        # Adam update
        updates, opt_state = adam_transform.update(gradients, opt_state)

        # update parameters with step size
        parameters = jax.tree_util.tree_map(
            lambda p, s, u: p + s * u, parameters, step_sizes, updates
        )

        # return updated parameters, current value, and optimizer state
        return parameters, value, opt_state, gradients

    return Optimizer(init, step)
