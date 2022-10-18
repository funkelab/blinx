from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap import fit
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':

    # generate a test trace
    y_test = 5
    seed = 100
    e_params = EmissionParams(mu_i=50, sigma_i=0.03, mu_b=200, sigma_b=0.15)
    t_model_t = TraceModel(e_params, p_on=0.05, p_off=0.05)
    x_trace, states = t_model_t.generate_trace(y_test, seed=seed,
                                               num_frames=4000)
  
    plt.plot(x_trace[:1000])
    plt.show()

    max_likelihood = None
    best_y = None

    # Calc likelyhood that trace arrose from different y values
    #ys = np.arange(3, 11)
    ys = [5]
    start = time.time()
    for y in ys:
        likelihood, p_on, p_off, mu, sigma = fit.optimize_params(y, x_trace)

        print('- '*20)
        print(f'y = {y}')
        print(f'log likelihood   = { likelihood:.2f}')
        print(f'p_on / p_off     = { p_on:.4f} / {p_off:.4f}')
        print(f'mu / sigma       = {mu:.4f} / {sigma:.4f}')

        if max_likelihood is None or likelihood > max_likelihood:
            max_likelihood = likelihood
            best_y = y

    print('* '*20)
    print(f'tested {len(ys)} y values in {time.time()-start:.2f}s')
    print(f'maximum likelihood y = {best_y}')
