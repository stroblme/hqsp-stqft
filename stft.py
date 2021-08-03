from IPython import get_ipython

import numpy as np
from numpy import array, pi

from math import log2

import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

from fft import fft_framework
from stt import stt_framework

class stft_framework():
    def __init__(self, transform=fft_framework):
        self.stt_inst = stt_framework(transform)

    def transform(self, y_signal, **kwargs):
        return self.stt_inst.stt_transform(y_signal, **kwargs)

    def postProcess(self, y_hat, f, t):
        return self.stt_inst.postProcess(y_hat, f, t)
        
        # y_split_list = y_signal.split(nSamplesWindow, overlapFactor=overlapFactor, windowType=windowType)
        # # y_split_list = y_signal.split(nSamplesWindow)
        # nParts = len(y_split_list)

        # y_hat = np.empty((self.fftInst.estimateSize(y_split_list[0]),nParts), dtype=np.complex64)
        # print(f"Transformation output will be of shape {y_hat.shape}")
        # for i in range(0, nParts):
        #     y_hat[:, i] = self.fftInst.transform(y_split_list[i])
        #     # spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        #     # autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        #     # result[i, :] = autopower[:fft_size]               # append to the results array
        
        # # result = 20*np.log10(result)          # scale to db
        # # result = np.clip(result, -40, 200)    # clip values

        # return y_hat