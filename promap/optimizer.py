from collections import namedtuple
from .parameters import Parameters
import optax


Optimizer = namedtuple("Optimizer", ["init", "step"])


def create_optimizer(value_grad_func, hyper_parameters):
    param_labels = ("MU", "Others")
    optimizer = optax.multi_transform(
        {
            "MU": optax.adam(hyper_parameters.mu_gradient_step_size),
            "Others": optax.adam(hyper_parameters.gradient_step_size),
        },
        param_labels,
    )

    def split_parameters(parameters):
        return (
            [parameters.mu, parameters.mu_bg],
            [parameters.sigma, parameters.p_on, parameters.p_off],
        )

    def join_parameters(params):
        mus, others = params
        mu, mu_bg = mus[0], mus[1]
        sigma, p_on, p_off = others[0], others[1], others[2]
        return Parameters(mu=mu, mu_bg=mu_bg, sigma=sigma, p_on=p_on, p_off=p_off)

    def init(parameters):
        params = split_parameters(parameters)
        return optimizer.init(params)

    def step(trace, parameters, opt_state):
        # split parameters into tuple form so that labels apply
        params = split_parameters(parameters)

        # get value and gradient

        value, gradients = value_grad_func(trace, parameters)

        # we maximize -> minimize with inverted gradients
        gradients = -gradients

        # gradients must also be in same tuple form
        grads = split_parameters(gradients)

        # compute updates from gradients
        updates, opt_state = optimizer.update(grads, opt_state)

        # update parameters
        params = optax.apply_updates(params, updates)

        # re-combine params into a Parameters object
        parameters = join_parameters(params)

        # return updated parameters, current value, and optimizer state

        return parameters, value, opt_state

    return Optimizer(init, step)
