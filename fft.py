from IPython import get_ipython

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

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

    def fft_recurse(self, y):
        N = len(y)
        
        if N == 1:
            y_hat = y
        else:
            # Get every second sample starting from zero and one (even and odd samples)
            y_hat_even = self.fft_recurse(y[::2])
            y_hat_odd = self.fft_recurse(y[1::2])

            # Fourier factor
            factor = np.exp(-2j*np.pi*np.arange(N)/ N)
            
            y_hat = np.concatenate([y_hat_even+factor[:int(N/2)]*y_hat_odd,
                                    y_hat_even+factor[int(N/2):]*y_hat_odd])

        return y_hat
