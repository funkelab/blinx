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

    def __init__(self, p_on=0.9, μ=1.0, σ=0.1, σ_background=0.1, q=0):

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

            y (ndarray, int, shape (n,) or (m, n)):

                Number of amino acids, congruent with x. If a 2D array is
                given, p(x|y) is computed for each row in y and an array of
                probabilities is returned.
        '''

        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

        amino_acids = np.nonzero(x >= 0)[0]
        logger.debug("Found measurements for amino acids %s", amino_acids)

        p = np.ones((y.shape[0],)) if len(y.shape) == 2 else 1
        for i in amino_acids:

            p_x_i_given_y_i = np.zeros((y.shape[0],)) \
                    if len(y.shape) == 2 else 0

            y_i = y[:,i] if len(y.shape) == 2 else y[i]
            max_y_i = np.max(y_i)

            logger.debug(
                "Amino acid %s occurs at most %d times",
                i, max_y_i)

            for z_i in range(max_y_i + 1):
                p_x_i_given_y_i += \
                    self.p_x_i_given_z_i(x[i], z_i) * \
                    self.p_z_i_given_y_i(z_i, y_i)

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

        p = self.p_on**z_i * (1.0 - self.p_on)**(y_i - z_i)
        p *= z_i <= y_i

        return p

