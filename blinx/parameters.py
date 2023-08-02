import jax
from jax.tree_util import register_pytree_node_class


def inv_sigmoid(p):
    return -jax.lax.log(1.0 / p - 1.0)


@register_pytree_node_class
class Parameters:
    """Contains all the parameters fit to the observed intensity trace.

    Args:

        mu (float):

            The mean intensity of a single 'on' emitter

        mu_bg (float):

            the mean background intensity, or the expected intensity
            when no emitters are 'on'

        sigma (float):

            the standard deviation in the intensity of a single 'on' emitter

        p_on (float):

            the probability of an emitter that is 'off' at time t-1 turning
            'on' at time t

        p_off (float):

            the probability of an emitter that is 'on' at time t-1 turning
            'off' at time t

        probs_are_logits (bool):

            Set to true to indicate that `p_on` and `p_off` are given as logits
            instead of as probabilities. Used internally.

    """

    def __init__(self, mu, mu_bg, sigma, p_on, p_off, probs_are_logits=False):
        self.mu = mu
        self.mu_bg = mu_bg
        self.sigma = sigma
        if probs_are_logits:
            self._p_on_logit = p_on
            self._p_off_logit = p_off
        else:
            self._p_on_logit = inv_sigmoid(p_on)
            self._p_off_logit = inv_sigmoid(p_off)

    @property
    def p_on(self):
        return jax.nn.sigmoid(self._p_on_logit)

    @property
    def p_off(self):
        return jax.nn.sigmoid(self._p_off_logit)

    def reshape(self, shape):
        """Reshape the tensors for each parameter to the given `shape`."""

        return Parameters(
            self.mu.reshape(shape),
            self.mu_bg.reshape(shape),
            self.sigma.reshape(shape),
            self._p_on_logit.reshape(shape),
            self._p_off_logit.reshape(shape),
            probs_are_logits=True,
        )

    def __getitem__(self, key):
        return Parameters(
            self.mu[key],
            self.mu_bg[key],
            self.sigma[key],
            self._p_on_logit[key],
            self._p_off_logit[key],
            probs_are_logits=True,
        )

    def __repr__(self):
        return (
            f"μ={self.mu}\t"
            f"μ_bg={self.mu_bg}\t"
            f"o={self.sigma}\t"
            f"p_on={self.p_on}\t"
            f"p_on logits={self._p_on_logit}\t"
            f"p_off logits={self._p_off_logit}"
        )

    def tree_flatten(self):
        children = (
            self.mu,
            self.mu_bg,
            self.sigma,
            self._p_on_logit,
            self._p_off_logit,
        )
        aux = None
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, probs_are_logits=True)
