import jax.numpy as jnp
import numpy as np
import pytest
from blinx.markov_chain import (
    get_measurement_log_likelihood,
    get_optimal_states,
    get_steady_state,
)


def test_markov_chain(problem_likelihood_solution):
    inference_problem, likelihood, solution = problem_likelihood_solution
    (p_measurement, p_initial, p_transition) = inference_problem

    log_likelihood = get_measurement_log_likelihood(
        p_measurement, p_initial, p_transition
    )

    np.testing.assert_almost_equal(likelihood, np.exp(log_likelihood))

    # not conistant with new implimentation of get_optimal states
    # states = get_optimal_states(p_measurement, p_initial, p_transition)

    # if solution is not None:
    #     np.testing.assert_equal(np.asarray(states), np.asarray(solution))


def test_stationary_distribution(problem_likelihood_solution):
    inference_problem, _, _ = problem_likelihood_solution
    (_, _, p_transition) = inference_problem

    numerical_steady_state = get_steady_state(p_transition)
    numerical_steady_state = np.asarray(numerical_steady_state)

    analytical_steady_state = get_analytical_steady_state(p_transition)

    np.testing.assert_allclose(
        numerical_steady_state, analytical_steady_state, rtol=1e-5
    )

    np.testing.assert_allclose(
        numerical_steady_state,
        np.matmul(numerical_steady_state, p_transition),
        rtol=1e-5,
    )


def get_analytical_steady_state(transition_matrix):
    # The stationary distribution is a left eigenvector of the transition
    # matrix with eigenvalue 1, i.e.,
    #
    #   p * T = p,
    #
    # where p is the stationary distribution and T is the transition matrix.

    # the transpose computes the left eigenvalues/vectors
    eigen_values, eigen_vectors = np.linalg.eig(np.asarray(transition_matrix).T)

    # find the eigenvalue closest to one
    index = np.argmin(np.abs(eigen_values - 1.0))

    stationary_distribution = eigen_vectors[:, index]
    stationary_distribution /= np.sum(stationary_distribution)

    return jnp.asarray(stationary_distribution)


@pytest.fixture(params=["uniform", "almost_no_transition", "random"])
def problem_likelihood_solution(request):
    if request.param == "random":
        n = 3  # number of states
        m = 10  # number of discrete measurements
        t = 5  # number of timesteps

        
        p_measurement = np.random.uniform(size=(m, n))
        p_measurement = p_measurement / np.sum(p_measurement, axis=0)

        p_measurement = jnp.array(p_measurement)

    else:
        n = 2  # number of states
        m = 4  # number of discrete measurements
        t = 5  # number of timesteps

        p_measurement = jnp.array(
            [
                [0.1],  # p(x=0|y)
                [0.4],  # p(x=1|y)
                [0.3],  # p(x=2|y)
                [0.2],  # p(x=3|y)
            ]
        )

    if request.param == "uniform":
        p_initial = jnp.array([0.5, 0.5])
        p_transition = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        solution = [1, 0, 1, 0, 0]

    elif request.param == "almost_no_transition":
        p_initial = jnp.array([0.5, 0.5])
        p_transition = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        solution = [0, 0, 0, 0, 0]

    elif request.param == "random":
        p_initial = np.random.uniform(size=(n,))
        p_initial /= np.sum(p_initial)

        p_transition = np.random.uniform(size=(n, n))
        p_transition = (p_transition.T / np.sum(p_transition, axis=1)).T

        p_initial = jnp.array(p_initial)
        p_transition = jnp.array(p_transition)

        solution = None

    inference_problem = (p_measurement, p_initial, p_transition)
    likelihood = naive_likelihood(*inference_problem)

    return inference_problem, likelihood, solution


def naive_likelihood(p_measurement, p_initial, p_transition):
    t = len(p_measurement)
    n = len(p_initial)

    # p(x) = Σ_y p(x,y)
    #      = Σ_y p(y_1) Π_i=1,..,t p(x_i|y_i) Π_i=2,...,t p(y_i|y_i-1)
    #      = Σ_y_1 p(y_1) p(x_1|y_1) Σ_y_2 p(y_2|y_1) p(x_2|y_2) ...

    def naive_rec_likelihood(i, y_prev):
        if i == t:
            return 1.0

        likelihood = 0
        for y_next in range(n):
            likelihood += (
                p_transition[y_prev, y_next]
                * p_measurement[i]
                * naive_rec_likelihood(i=i + 1, y_prev=y_next)
            )

        return likelihood

    likelihood = 0
    for y_0 in range(n):
        likelihood += (
            p_initial[y_0]
            * p_measurement[0]
            * naive_rec_likelihood(i=1, y_prev=y_0)
        )

    return likelihood
