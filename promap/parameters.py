from collections import namedtuple


class Parameters(namedtuple(
        "Parameters",
        ["mu", "mu_bg", "sigma", "p_on", "p_off"])):

    def __neg__(self):
        return Parameters(*(-p for p in self))
