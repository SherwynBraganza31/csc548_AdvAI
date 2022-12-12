import numpy as np
import sys
import itertools


class FourierBasis:
    def __init__(self, order: int, dimensions: int):
        # Instance variables
        self.coefficients = np.array([])
        self.gradient_factors = np.array([])
        self.dimensions = dimensions
        self.order = [order] * self.dimensions

        # create empty container for coefficient array
        prods = [range(0, o + 1) for o in self.order]
        coeffs = [v for v in itertools.product(*prods)]
        self.coefficients = np.array(coeffs)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.gradient_factors = 1.0 / np.linalg.norm(self.coefficients, ord=2, axis=1)
        self.gradient_factors[0] = 1.0  # Overwrite division by zero for function with all-zero coefficients.

    def getFourierBasisApprox(self, state_vector: np.ndarray):
        """
        Computes basis function values at a given state.
        """

        # Bounds check state vector
        if  np.min(state_vector) < 0.0 or np.max(state_vector) > 1.0:
            print('Fourier Basis: Given State Vector ({}) not in range [0.0, 1.0]'.format(state_vector),
                  file=sys.stderr)

        # Compute the Fourier Basis feature values
        return np.cos(np.pi * np.dot(self.coefficients, state_vector))

    def getShape(self):
        return self.coefficients.shape

    def getGradientFactors(self):
        return self.gradient_factors

    def getGradientFactor(self, function_no):
        return self.gradient_factors[function_no]

    def length(self):
        """Return the number of basis functions."""
        return self.coefficients.shape[0]

# def RadialBasis: