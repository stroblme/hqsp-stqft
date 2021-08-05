from IPython import get_ipython

import numpy as np
from numpy import array, pi, sign

from math import log2

from frontend import signal

import matplotlib.pyplot as plt 
from numpy.core.fromnumeric import size

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

class stt_framework():
    def __init__(self, transformation, **kwargs):
        self.transformationInst = transformation(**kwargs)

    def stt_transform(self, y_signal, nSamplesWindow=2**10, overlapFactor=0, windowType=None, suppressPrint=False):
        
        y_split_list = y_signal.split(nSamplesWindow, overlapFactor=overlapFactor, windowType=windowType)
        # y_split_list = y_signal.split(nSamplesWindow)
        nParts = len(y_split_list)

        y_hat = np.empty((self.transformationInst.estimateSize(y_split_list[0]),nParts), dtype=np.complex64)
        if not suppressPrint:
            print(f"Transformation output will be of shape {y_hat.shape}")

        for i in range(0, nParts):
            if not suppressPrint:
                print(f"Running iteration {i} of {nParts}")
            y_hat[:, i] = self.transformationInst.transform(y_split_list[i])
            # spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
            # autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
            # result[i, :] = autopower[:fft_size]               # append to the results array
        
        # result = 20*np.log10(result)          # scale to db
        # result = np.clip(result, -40, 200)    # clip values

        return y_hat

    def stt_transformInv(self, y_signal, nSamplesWindow=2**10, overlapFactor=0, windowType=None, suppressPrint=False):
        hopSize = np.int32(np.floor(nSamplesWindow * (1-overlapFactor)))
        # nParts = np.int32(np.ceil(len(y_signal.t) / np.float32(hopSize)))
        
        y_hat = np.zeros((len(y_signal.t)+1)*hopSize, dtype=np.float64)
        y_signal_part = signal()

        nParts = len(y_signal.t)
        for i in range(0,nParts):
            if not suppressPrint:
                print(f"Running iteration {i} of {nParts}")

            y_signal_part.externalSample(y_signal.y[:,i], y_signal.t)

            pt = i*hopSize
            # if i == 0:
            #---a
            # y_hat[pt:pt+int(len(y_signal.f)*overlapFactor)] += np.float64(self.transformationInst.transform(y_signal_part))            # y_hat_temp = np.float64(self.transformationInst.transform(y_signal_part))
            #---

            #---b
            # y_hat_temp = np.float64(self.transformationInst.transform(y_signal_part))

            # if i == 0:
            #     y_hat[:len(y_signal.f)] += y_hat_temp
            # elif i == nParts-1:
            #     y_hat[-len(y_signal.f):] += y_hat_temp
            # else:
            #     y_hat[int(pt-hopSize/2):int(pt+len(y_signal.f)*overlapFactor+hopSize/2)] += y_hat_temp
            #---

            #---c
            y_hat_temp = np.float64(self.transformationInst.transform(y_signal_part))

            n = y_hat_temp.shape[0]//2
            y_hat_sliced =y_hat_temp[n:]/n 

            y_hat[pt:pt+int(len(y_signal.f)*overlapFactor)] += y_hat_sliced

            #---
        return y_hat

    def postProcess(self, y_hat, f, t):
        for t_idx in range(0, y_hat.shape[1]):
            y_hat[:,t_idx], f= self.transformationInst.postProcess(y_hat[:,t_idx], f)

        return y_hat, f, t