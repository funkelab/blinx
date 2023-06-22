from collections import namedtuple


class Parameters(namedtuple("Parameters", ["mu", "mu_bg", "sigma", "p_on", "p_off"])):
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

    """

    def __neg__(self):
        # needed for the computation of gradients through the class
        return Parameters(*(-p for p in self))
