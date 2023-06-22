from blinx import HyperParameters, ParameterRanges
from blinx.parameters import Parameters
from blinx.trace_model import generate_trace
from blinx.estimate import estimate_y, get_initial_parameter_guesses, optimize_parameters
from blinx.trace_model import get_trace_log_likelihood
from blinx.optimizer import create_optimizer
import jax.numpy as jnp
import jax
import time

def create_traces(y, num_traces, num_frames):
    parameters = Parameters(mu=3000.0, mu_bg=5000.0, sigma=0.03, p_on=0.05, p_off=0.05)
    traces = []
    for seed in range(num_traces):
        trace, zs = generate_trace(y, parameters, num_frames)
        traces.append(trace)

    return jnp.array(traces)

def benchmark_num_traces():
    y = 5
    traces = create_traces(y, 10, 4000)
    
    parameter_ranges = ParameterRanges(
        mu_range=(3000, 4000),
        mu_bg_range=(5000, 5000),
        sigma_range=(0.03, 0.03),
        p_on_range=(1e-3, 0.1),
        p_off_range=(1e-3, 0.1),
        mu_step=20,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=5,
        p_off_step=5
        )
    
    hyper_parameters = HyperParameters(
        min_y= y - 1,
        num_guesses=3,
        epoch_length=1000,
        is_done_limit=1e-5,
        step_sizes=Parameters(mu=1e-1, mu_bg=1e-1, sigma=1e-8, p_on=1e-6, p_off=1e-6),
        distribution_threshold=1e-1,
        max_x=y * 3000 * 2,
        num_x_bins=1024,
        p_outlier=0.1,
        )
    start_1 = time.time()
    y, parameters, likelihoods = estimate_y(
        traces=create_traces(5, 1, 4000),
        max_y= y + 1,
        parameter_ranges=parameter_ranges,
        hyper_parameters=hyper_parameters)
    print(f'1 trace fit in {time.time()-start_1:.3f}s')
    
    start_2 = time.time()
    y=5
    y, parameters, likelihoods = estimate_y(
        traces=create_traces(5, 2, 4000),
        max_y= y + 1,
        parameter_ranges=parameter_ranges,
        hyper_parameters=hyper_parameters)
    print(f'2 traces fit in {time.time()-start_2:.3f}s')
    
    start_5 = time.time()
    y=5
    y, parameters, likelihoods = estimate_y(
        traces=create_traces(5, 5, 4000),
        max_y= y + 1,
        parameter_ranges=parameter_ranges,
        hyper_parameters=hyper_parameters)
    print(f'5 traces fit in {time.time()-start_5:.3f}s')

def benchmark_initial_guesses():
    y=5

    parameter_ranges = ParameterRanges(
        mu_range=(3000, 4000),
        mu_bg_range=(5000, 5000),
        sigma_range=(0.03, 0.03),
        p_on_range=(1e-3, 0.1),
        p_off_range=(1e-3, 0.1),
        mu_step=20,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=5,
        p_off_step=5
        )
    
    hyper_parameters = HyperParameters(
        min_y= y - 1,
        num_guesses=3,
        epoch_length=1000,
        is_done_limit=1e-5,
        step_sizes=Parameters(mu=1e-1, mu_bg=1e-1, sigma=1e-8, p_on=1e-6, p_off=1e-6),
        distribution_threshold=1e-1,
        max_x=y * 3000 * 2,
        num_x_bins=1024,
        p_outlier=0.1,
        )
    traces = create_traces(y, 1, 4000)
    start_1 = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    print(f'found_initial_guesses for 1 trace in {time.time()-start_1:.4f}s')
    
    traces = create_traces(y, 2, 4000)
    start_2 = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    print(f'found_initial_guesses for 2 traces in {time.time()-start_2:.4f}s')
    
    traces = create_traces(y, 10, 4000)
    start_10 = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    print(f'found_initial_guesses for 10 traces in {time.time()-start_10:.4f}s')
    
    traces = create_traces(y, 50, 4000)
    start_50 = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    print(f'found_initial_guesses for 50 traces in {time.time()-start_50:.4f}s')

def temp_fit_epoch(y, traces, parameter_ranges, hyper_parameters):
    start = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    log_likelihood_grad_func = jax.value_and_grad(
        lambda t, p: get_trace_log_likelihood(t, y, p, hyper_parameters), argnums=1
    )

    optimizer = create_optimizer(log_likelihood_grad_func, hyper_parameters)

    optimizer_states = jax.vmap(jax.vmap(optimizer.init))(parameter_guesses)


    parameters = parameter_guesses

    fit_epoch = jax.vmap(  # vmap over traces
        jax.vmap(  # vmap over parameters
            lambda t, p, os: optimize_parameters(t, p, os, optimizer, hyper_parameters),
            in_axes=(None, 0, 0),
        )
    )
    fit_epoch = jax.jit(fit_epoch)
    parameters, optimizer_states, log_likelihoods, is_done = fit_epoch(
        traces, parameters, optimizer_states
    )
    print(f'ran {traces.shape[0]} traces in {time.time()-start:.3f}s')
    return 

def temp_fit_epoch_2(y, traces, parameter_ranges, hyper_parameters):
    start = time.time()
    parameter_guesses = get_initial_parameter_guesses(
        traces, y, parameter_ranges, hyper_parameters
    )
    log_likelihood_grad_func = jax.value_and_grad(
        lambda t, p: get_trace_log_likelihood(t, y, p, hyper_parameters), argnums=1
    )

    optimizer = create_optimizer(log_likelihood_grad_func, hyper_parameters)

    optimizer_states = jax.vmap(jax.vmap(optimizer.init))(parameter_guesses)


    parameters = parameter_guesses

    fit_epoch = jax.vmap(  # vmap over traces
        jax.vmap(  # vmap over parameters
            lambda t, p, os: optimize_parameters(t, p, os, optimizer, hyper_parameters),
            in_axes=(None, 0, 0),
        ), in_axes=(0,0,0)
    )
    fit_epoch = jax.jit(fit_epoch)
    parameters, optimizer_states, log_likelihoods, is_done = fit_epoch(
        traces, parameters, optimizer_states
    )
    print(f'ran {traces.shape[0]} traces in {time.time()-start:.3f}s')
    return 


def benchmark_optimization():
    y=5
    parameter_ranges = ParameterRanges(
        mu_range=(3000, 3100),
        mu_bg_range=(5000, 5000),
        sigma_range=(0.03, 0.03),
        p_on_range=(1e-3, 0.1),
        p_off_range=(1e-3, 0.1),
        mu_step=5,
        mu_bg_step=1,
        sigma_step=1,
        p_on_step=5,
        p_off_step=5
        )
    
    hyper_parameters = HyperParameters(
        min_y= y - 1,
        num_guesses=1,
        epoch_length=1000,
        is_done_limit=1e-5,
        step_sizes=Parameters(mu=1e-1, mu_bg=1e-1, sigma=1e-8, p_on=1e-6, p_off=1e-6),
        distribution_threshold=1e-1,
        max_x=y * 3000 * 2,
        num_x_bins=1024,
        p_outlier=0.1,
        )
    
    traces = create_traces(y, 1, 4000)
    a = temp_fit_epoch(y, traces, parameter_ranges, hyper_parameters)
    #b = temp_fit_epoch_2(y, traces, parameter_ranges, hyper_parameters)
    
    traces_2 = create_traces(y, 2, 4000)
    a = temp_fit_epoch(y, traces_2, parameter_ranges, hyper_parameters)
    #c = temp_fit_epoch_2(y, traces_2, parameter_ranges, hyper_parameters)
    
    traces_3 = create_traces(y, 3, 4000)
    a = temp_fit_epoch(y, traces_3, parameter_ranges, hyper_parameters)
    #c = temp_fit_epoch_2(y, traces_2, parameter_ranges, hyper_parameters)
    
    # traces_50 = create_traces(y, 50, 4000)
    # a = temp_fit_epoch(y, traces_50, parameter_ranges, hyper_parameters)
    
    # traces_100 = create_traces(y, 100, 4000)
    # a = temp_fit_epoch(y, traces_100, parameter_ranges, hyper_parameters)
    
    return
if __name__ == '__main__':
    benchmark_initial_guesses()
    benchmark_optimization()
    
    
    