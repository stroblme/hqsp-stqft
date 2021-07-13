from IPython import get_ipython

import numpy as np
from numpy import pi

class dft_framework():
    def transform(self, y_signal):
        """
        Function to calculate the 
        discrete Fourier Transform 
        of a 1D real-valued signal x
        """

        y = y_signal.sample()

        N = len(y)
        n = np.arange(N)
        k = n.reshape((N, 1))

        e = np.exp(-2j * np.pi * k * n / N)

        Y = np.dot(e, y)
        
        return Y

    def postProcess(self, y_hat, f):
        return y_hat, f