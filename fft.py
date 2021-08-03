from IPython import get_ipython

import numpy as np
from numpy import pi

from math import log2

from utils import isPow2

class fft_framework():
    def transform(self, y_signal):
        """
        A recursive implementation of 
        the 1D Cooley-Tukey FFT, the 
        input should have a length of 
        power of 2. 
        """
        y = y_signal.sample()
        y_hat = self.fft_recurse(y)

        return y_hat

    def transformInv(self, y_signal):
        y_hat = y_signal.sample()
        y = self.ifft_recurse(y_hat)

        return y

    def postProcess(self, y_hat, f):
        return y_hat, f

    def fft_recurse(self, y):
        N = len(y)
        
        if N == 1:
            y_hat = y
        else:
            # Get every second sample starting from zero and one (even and odd samples)
            y_hat_even = self.fft_recurse(y[::2])
            y_hat_odd = self.fft_recurse(y[1::2])

            # Fourier factor
            rotation = np.exp(-2j*np.pi*np.arange(N)/ N)
            
            y_hat = np.concatenate([y_hat_even+rotation[:int(N/2)]*y_hat_odd,
                                    y_hat_even+rotation[int(N/2):]*y_hat_odd])

        return y_hat

    def ifft_recurse(self, y):
        N = len(y)
        
        if N == 1:
            y_hat = y
        else:
            # Get every second sample starting from zero and one (even and odd samples)
            y_hat_even = self.fft_recurse(y[::2])
            y_hat_odd = self.fft_recurse(y[1::2])

            # Fourier factor
            rotation = np.exp(+2j*np.pi*np.arange(N)/ N)
            
            y_hat = np.concatenate([y_hat_even+rotation[:int(N/2)]*y_hat_odd,
                                    y_hat_even+rotation[int(N/2):]*y_hat_odd])

        return y_hat

    def estimateSize(self, y_signal):
        assert isPow2(y_signal.nSamples)

        n_bins = int((log2(y_signal.nSamples)/log2(2)))

        return 2**n_bins