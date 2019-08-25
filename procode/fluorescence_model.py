import numpy as np
import logging

logger = logging.getLogger(__name__)


class FluorescenceModel:
    '''Simple fluorescence model for amino acid counts in proteins (y), dye
    activity (z), and fluorescence measurements per dye (x)::

        y_i ∈ ℕ (count of amino acid i)
        z_i ∈ ℕ (number of active dyes for amino acid i)
        x_i ∈ ℝ (total flourescence of active dyes for amino acid i)

        # independent per dye
        p(x|y) = Σ_i p(x_i|y_i)

        # fluorescence depends on dye activity z_i
        p(x_i|y_i) = Σ_z_i p(x_i|z_i)p(z_i|y_i)

        # dye activity is binomial
        p(z_i|y_i) ~ B(y_i, p_on)

        # flourescence follows log-normal distribution
        p(x*_i|z_i) = sqrt(2πσ_i²)^-1 exp[ -(x*_i - μ_i - ln z_i + q_z_i)² ]

    Args:

        p_on:
            Probability of a dye to be active.

        μ:
            Mean log intensity of a dye.

        σ:
            Standard deviation of log intensity of a dye.

        σ_background:
            Standard deviation of log intensity of background (mean is assumed
            to be 0).

        q:
            Dye-dye interaction factor (see Mutch et al., Biophusical Journal,
            2007, Deconvolving Single-Molecule Intensity Distributions for
            Quantitative Microscopy Measurements)
    '''

    def __init__(self, p_on, μ, σ, σ_background, q):

        self.p_on = p_on
        self.μ = μ
        self.σ2 = σ**2
        self.σ2_background = σ_background**2
        self.q = q

    def p_x_given_y(self, x, y):
        '''Compute p(x|y).

        Args:

            x (ndarray, float, shape (n,)):

                Measured fluorescences per dye. -1 to indicate no measurement.

            y (ndarray, int, shape (n,)):

                Number of amino acids, congruent with x.
        '''

        amino_acids = np.nonzero(x >= 0)[0]
        logger.debug("Found measurements for amino acids %s", amino_acids)

        p = 1
        for i in amino_acids:

            p_x_i_given_y_i = 0

            logger.debug(
                "Amino acid %s occurs %d times in protein %s",
                i, y[i], y)

            for z_i in range(y[i]):
                p_x_i_given_y_i += \
                    self.p_x_i_given_z_i(x[i], z_i) * \
                    self.p_z_i_given_y_i(z_i, y[i])

            p *= p_x_i_given_y_i

        return p

    def p_x_i_given_z_i(self, x_i, z_i):

        if z_i == 0:
            μ = 0
            σ2 = self.σ2_background
        else:
            μ = self.μ + np.log(z_i) - self.q
            σ2 = self.σ2

        return self.log_normal(x_i, μ, σ2)

    def log_normal(self, x, μ, σ2):
        return \
            1.0/(x*np.sqrt(2.0*np.pi*σ2))* \
            np.exp(-(np.log(x) - μ)**2/(2.0*σ2))

    def p_z_i_given_y_i(self, z_i, y_i):

        if z_i > y_i:
            return 0.0

        return self.p_on**z_i * (1.0 - self.p_on)**(y_i - z_i)
