import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .fluorescence_model import p_norm


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

    def __init__(
        self, r_e, r_bg, mu_ro, sigma_ro, gain, p_on, p_off, probs_are_logits=False
    ):
        self.r_e = r_e
        self.r_bg = r_bg
        self.mu_ro = mu_ro
        self.sigma_ro = sigma_ro
        self.gain = gain
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
            self.r_e.reshape(shape),
            self.r_bg.reshape(shape),
            self.mu_ro.reshape(shape),
            self.sigma_ro.reshape(shape),
            self.gain.reshape(shape),
            self._p_on_logit.reshape(shape),
            self._p_off_logit.reshape(shape),
            probs_are_logits=True,
        )

    def flatten(self):
        """Convert this class into just a single tensor."""

        return jnp.array(
            [
                Parameters._flatten_rec(self.r_e),
                Parameters._flatten_rec(self.r_bg),
                Parameters._flatten_rec(self.mu_ro),
                Parameters._flatten_rec(self.sigma_ro),
                Parameters._flatten_rec(self.gain),
                Parameters._flatten_rec(self._p_on_logit),
                Parameters._flatten_rec(self._p_off_logit),
            ]
        )

    @staticmethod
    def _flatten_rec(parameters):
        if isinstance(parameters, Parameters):
            return parameters.flatten()
        else:
            return parameters

    def __getitem__(self, key):
        return Parameters(
            self.r_e[key],
            self.r_bg[key],
            self.mu_ro[key],
            self.sigma_ro[key],
            self.gain[key],
            self._p_on_logit[key],
            self._p_off_logit[key],
            probs_are_logits=True,
        )

    def __repr__(self):
        return (
            f"r_e={self.r_e}\t"
            f"r_bg={self.r_bg}\t"
            f"μ_ro={self.mu_ro}\t"
            f"o_ro={self.sigma_ro}\t"
            # f"p_on={self.p_on}\t"
            # f"p_off={self.p_off}\t"
            f"gain={self.gain}\t"
            f"p_on logits={self._p_on_logit}\t"
            f"p_off logits={self._p_off_logit}"
        )

    def tree_flatten(self):
        children = (
            self.r_e,
            self.r_bg,
            self.mu_ro,
            self.sigma_ro,
            self.gain,
            self._p_on_logit,
            self._p_off_logit,
        )
        aux = None
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, probs_are_logits=True)

    @classmethod
    def stack(cls, parameters):
        return Parameters(
            jnp.stack([p.r_e for p in parameters]),
            jnp.stack([p.r_bg for p in parameters]),
            jnp.stack([p.mu_ro for p in parameters]),
            jnp.stack([p.sigma_ro for p in parameters]),
            jnp.stack([p.gain for p in parameters]),
            jnp.stack([p._p_on_logit for p in parameters]),
            jnp.stack([p._p_off_logit for p in parameters]),
            probs_are_logits=True,
        )
