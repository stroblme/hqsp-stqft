# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
# get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Classical (D)FT
# 
# First let's do a classical dft on a toy-signal to get familiar again with this whole signal processing topic

# %%

class dft_framework():
    def transform(self, x):
        """
        Function to calculate the 
        discrete Fourier Transform 
        of a 1D real-valued signal x
        """

        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        e = np.exp(-2j * np.pi * k * n / N)
        
        X = np.dot(e, x)
        
        return X





# # %%
# X = DFT(x)

# # calculate the frequency
# N = len(X)
# n = np.arange(N)
# T = N/sr
# freq = n/T 


# # %%
# n_oneside = N//2
# # get the one side frequency
# f_oneside = freq[:n_oneside]

# # normalize the amplitude
# X_oneside =X[:n_oneside]/n_oneside

# plt.figure(figsize = (12, 6))
# plt.subplot(121)
# plt.stem(f_oneside, abs(X_oneside), 'b',          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('DFT Amplitude |X(freq)|')

# plt.subplot(122)
# plt.stem(f_oneside, abs(X_oneside), 'b',          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.xlim(0, 10)
# plt.tight_layout()
# plt.show()