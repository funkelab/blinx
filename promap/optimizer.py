from collections import namedtuple
import optax
import jax.numpy as jnp


Optimizer = namedtuple('Optimizer', ['init', 'step'])


def create_optimizer(value_grad_func, hyper_parameters):

    param_labels = ('MU', 'Others')
    optimizer = optax.multi_transform(
        {'MU': optax.adam(hyper_parameters.mu_gradient_step_size),
         'Others': optax.adam(hyper_parameters.gradient_step_size)},
        param_labels)

    def init(parameters):
        # parameters must be in tuple form so that labels defined above apply
        params = (parameters[:2], parameters[2:])

        return optimizer.init(params)

    def step(trace, parameters, opt_state):

        # split parameters into tuple form so that labels apply
        params = (parameters[:2], parameters[2:])

        # get value and gradient

        value, gradients = value_grad_func(trace, parameters)

        # we maximize -> minimize with inverted gradients
        gradients = -gradients

        # gradients must also be in same tuple form
        grads = (gradients[:2], gradients[2:])

        # compute updates from gradients
        updates, opt_state = optimizer.update(
            grads,
            opt_state)

        # update parameters
        params = optax.apply_updates(params, updates)

        # re-stack parametes into a jax-array
        parameters = jnp.hstack((params[0], params[1]))

        # return updated parameters, current value, and optimizer state

        return parameters, value, opt_state

    return Optimizer(init, step)
