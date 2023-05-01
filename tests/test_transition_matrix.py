import jax.numpy as jnp
import numpy as np
import pytest
from blinx.trace_model import create_transition_matrix
from scipy import stats


@pytest.fixture(params=[1, 2, 3, 10])
def y(request):
    return request.param


@pytest.fixture(params=[1e-4, 1e-2, 0.1, 0.2, 0.4])
def p_on(request):
    return request.param


@pytest.fixture(params=[1e-4, 1e-2, 0.1, 0.2, 0.4])
def p_off(request):
    return request.param


@pytest.fixture
def true_transition_matrix(y, p_on, p_off):
    size = y + 1  # possible states range from 0 - y inclusive
    transition_m = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = 0
            for z in range(i + 1):
                p += stats.binom.pmf(z, i, p_off) * stats.binom.pmf(
                    j - i + z, y - i, p_on
                )
            transition_m[i, j] = p

    return (y, p_on, p_off), jnp.array(transition_m)


def test_consistency(true_transition_matrix):
    (y, p_on, p_off), _ = true_transition_matrix

    transition_matrix = create_transition_matrix(y, p_on, p_off)

    # check that there are no zeros in the transition_matrix
    assert transition_matrix.all()

    # check that all rows sum to 1
    row_sums = jnp.sum(transition_matrix, axis=1)
    np.testing.assert_allclose(np.asarray(row_sums), np.ones(y + 1), rtol=1e-6)

    # check that there are no NaNs
    assert not np.any(np.isnan(np.asarray(transition_matrix)))


def test_correctness(true_transition_matrix):
    (y, p_on, p_off), compare = true_transition_matrix

    transition_matrix = create_transition_matrix(y, p_on, p_off)

    # we know that for very small p_on/p_off, the straight forward computation
    # differs numerically more from our implementation (and that's okay)
    if min(p_on, p_off) < 1e-2:
        rtol = 1e-4
    else:
        rtol = 1e-6

    np.testing.assert_allclose(
        np.asarray(transition_matrix), np.asarray(compare), rtol=rtol
    )
