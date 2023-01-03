import jax.numpy as jnp
import jax
from jax import lax
from promap.trace_model import TraceModel
from promap.fluorescence_model import FluorescenceModel
from promap import transition_matrix
from promap.constants import P_ON, P_OFF, MU
import optax


def optimize_params(y,
        trace,
        optimize_meth='joint_2_optimizer',
        initial_params=None,
        sigma_guess=0.2,
        mu_b_guess=5000,
        mu_lr=5):
    '''
    Fit kinetic (p_on / off) and emission (mu / sigma) parameters
    to an intensity trace for a given value of y

    Args:
        y (int):
            - The assumed total number of fluorescent emitters

        trace (jnp array):
            - ordered array of intensity observations
            - shape (number_observations, )

        initial_params (list of arrays) (float) or None:
            - initial guesses for p_on, p_off, and mu
            - format list([P_ON], [P_OFF], [MU])
            - if None, then will automatically find initial guesses

        optimize_meth (string):
            - specifies the optimizer to use for gradient descent

        sigma_guess (float):
            - initial guess for sigma
            - is the easiest param to fit so no need for multiple guesses

        mu_b_guess (float / int):
            - guess for background intensity value

        mu_lr (float):
            - the learning rate for the mu optimizer
            - needs to be individually set because of difference in magnitude
            between mu and other parameters

    Returns:
        The maximum log-likelihood that the trace arrose from y elements,
        as well as the optimum values of p_on, p_off, mu, and sigma
    '''

    # Define the loss function for grad descent
    bound_likelihood = lambda p_on, p_off, mu, sigma: _likelihood_func(y,
        p_on, p_off, mu, sigma, trace,
        mu_b_guess=mu_b_guess)
    grad_func = jax.jit(
        jax.value_and_grad(bound_likelihood, argnums=(0, 1, 2, 3)))

    # find initial guesses for the 4 parameters
    if initial_params is None:
        initial_params = _initial_guesses(mu_min=100, p_max=0.2, y=y,
              trace=trace, mu_b_guess=mu_b_guess)

    if optimize_meth == 'joint_2_optimizer':
        optimizer = _joint_2_optimizer

    num_local_minima = len(initial_params[P_ON])
    best_likelihood = None
    for i in range(num_local_minima):
        results = optimizer(p_on=initial_params[P_ON][i],
            p_off=initial_params[P_OFF][i],
            mu=initial_params[MU][i],
            sigma=sigma_guess,
            mu_lr=mu_lr, grad_func=grad_func)

        if best_likelihood is None or results[0] > best_likelihood:
            best_likelihood = results[0]
            best_params = results[1:]

    return best_likelihood, best_params


def _joint_2_optimizer(p_on, p_off, mu, sigma, mu_lr, grad_func):
    '''
    - optimize all 4 parameters jointly
    - seperate optimizer for mu, but all 4 params fit at the same time
    '''
    params = (p_on, p_off, mu, sigma)
    optimizer = optax.adam(learning_rate=1e-3, mu_dtype='uint64')
    opt_state = optimizer.init(params)

    mu_optimizer = optax.sgd(learning_rate=mu_lr)
    mu_opt_state = mu_optimizer.init(params[2])

    old_likelihood = 1
    diff = 10

    while diff > 1e-4:

        likelihood, grads = grad_func(p_on, p_off, mu, sigma)

        updates, opt_state = optimizer.update(grads, opt_state)

        mu_update, mu_opt_state = mu_optimizer.update(grads[2], mu_opt_state)

        p_on, p_off, _, sigma = optax.apply_updates((p_on, p_off, mu, sigma),
            updates)

        mu = optax.apply_updates((mu), mu_update)

        diff = jnp.abs(likelihood - old_likelihood)
        old_likelihood = likelihood

        # print(
        #     f'{likelihood:.2f}, {p_on:.4f}, {p_off:.4f}'
        #     f', {mu:.4f}, {sigma:.4f}')
        # print('-'*50)

    return -1*likelihood, p_on, p_off, mu, sigma


def _likelihood_func(y, p_on, p_off, mu, sigma, trace, mu_b_guess):
    '''
    Returns the likelihood of a trace given:
        a count (y),
        kinetic parameters (p_on & p_off), and
        emission parameters (mu & sigma)
    '''
    fluorescence_model = FluorescenceModel(
        mu_i=mu,
        sigma_i=sigma,
        mu_b=mu_b_guess,
        sigma_b=0.05)
    t_model = TraceModel(fluorescence_model)

    probs = t_model.fluorescence_model.p_x_given_zs(trace, y)

    comb_matrix = transition_matrix._create_comb_matrix(y)
    comb_matrix_slanted = transition_matrix._create_comb_matrix(
        y,
        slanted=True)

    def c_transition_matrix_2(p_on, p_off):
        return transition_matrix.create_transition_matrix(
            y, p_on, p_off,
            comb_matrix,
            comb_matrix_slanted)

    transition_mat = c_transition_matrix_2(p_on, p_off)
    p_initial = transition_matrix.p_initial(y, transition_mat)
    likelihood = t_model.get_likelihood(
        probs,
        transition_mat,
        p_initial)

    # need to flip to positive value for grad descent
    return -1 * likelihood


def _initial_guesses(mu_min, p_max, y, trace, mu_b_guess, sigma=0.05):
    '''
    Provides a rough estimate of parameters (p_on, p_off, and mu)
    Grid searches over defined parameter space and returns the minimum
    log likelihood parameters
    '''
    mus = jnp.linspace(mu_min, jnp.max(trace), 100)
    p_s = jnp.linspace(1e-4, p_max, 20)

    bound_likelihood = lambda mu, p_on, p_off: _likelihood_func(y, p_on,
        p_off, mu, sigma, trace, mu_b_guess)

    result = jax.vmap(jax.vmap(jax.vmap(bound_likelihood,
        in_axes=(0, None, None)),
        in_axes=(None, 0, None)),
        in_axes=(None, None, 0))(mus, p_s, p_s)

    minima_indecies = _find_minima_3d(result, 3)
    p_on_guess = p_s[minima_indecies[P_ON, :]]
    p_off_guess = p_s[minima_indecies[P_OFF, :]]
    mu_guess = mus[minima_indecies[MU, :]]
    likelihoods = result[tuple(minima_indecies)]

    return p_on_guess, p_off_guess, mu_guess, likelihoods


def _find_minima_3d(test_vec, window):
    '''
    - Finds the local minima of a 3D array
    - returns the minima indecies as an array of shape 3 x num_minima
    - for the first axis: 0 = p_on, 1 = p_off, 2 = mu
    '''
    mu_indecies = jnp.arange(test_vec.shape[2]) \
        [window:(test_vec.shape[2]-window)]
    p_off_indecies = jnp.arange(test_vec.shape[0]) \
        [window:(test_vec.shape[0]-window)]
    p_on_indecies = jnp.arange(test_vec.shape[0]) \
        [window:(test_vec.shape[0]-window)]

    def scan_func(vector, p_on_index, p_off_index, mu_index):
        vector_slice = lax.dynamic_slice(vector,
             (p_on_index-window, p_off_index-window, mu_index-window),
             (2*window, 2*window, 2*window))
        slice_min = jnp.min(vector_slice).astype('int32')
        all_same = jnp.all(vector_slice == vector_slice[0])
        b = jax.lax.cond(all_same, lambda: 0, lambda: slice_min)
        return b

    a = jax.vmap(jax.vmap(jax.vmap(scan_func,
                 in_axes=(None, None, None, 0)),
                 in_axes=(None, None, 0, None)),
                 in_axes=(None, 0, None, None))(test_vec, p_on_indecies,
                                                p_off_indecies, mu_indecies)

    a_pad = jnp.pad(a, window)
    local_minima = jnp.asarray(jnp.where(a_pad == test_vec))

    return local_minima
