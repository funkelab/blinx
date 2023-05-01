from collections import namedtuple

import jax

Optimizer = namedtuple("Optimizer", ["init", "step"])


def create_optimizer(value_grad_func, hyper_parameters):
    step_sizes = hyper_parameters.step_sizes

    def init(parameters):
        pass

    def step(trace, parameters, opt_state):
        # get value and gradient

        value, gradients = value_grad_func(trace, parameters)

        # update parameters (subtract to maximize)
        parameters = jax.tree_util.tree_map(
            lambda p, s, g: p - s * g, parameters, step_sizes, gradients
        )

        # return updated parameters, current value, and optimizer state
        return parameters, value, opt_state

    return Optimizer(init, step)
