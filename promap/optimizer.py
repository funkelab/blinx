from collections import namedtuple
import optax


Optimizer = namedtuple('Optimizer', ['init', 'step'])


def create_optimizer(value_grad_func, hyper_parameters):

    optimizer = optax.adam(
        learning_rate=hyper_parameters.gradient_step_size,
        mu_dtype='uint64')  # TODO: is that still needed?

    def init(parameters):

        return optimizer.init(parameters)

    def step(trace, parameters, opt_state):

        # get value and gradient

        value, gradients = value_grad_func(trace, parameters)

        # compute updates from gradients

        updates, opt_state = optimizer.update(
            gradients,
            opt_state)

        # update parameters

        parameters = optax.apply_updates(parameters, updates)

        # return updated parameters, current value, and optimizer state

        return parameters, value, opt_state

    return Optimizer(init, step)
