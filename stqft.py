from IPython import get_ipython

import numpy as np
from numpy import array, pi

from math import log2

import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

from qft import qft_framework


class stqft_framework():
    def __init__(self, numOfShots=1024):
        self.qftInst = qft_framework(numOfShots=numOfShots)

    def transform(self, y_signal, nSamplesWindow, overlapFactor=0, windowType=None):

        y_split_list = y_signal.split(nSamplesWindow, overlapFactor=overlapFactor, windowType=windowType)
        nParts = len(y_split_list)
        print(f"Signal divided into {nParts} parts")

        y_hat = np.empty((self.qftInst.estimateSize(y_split_list[0]),nParts),dtype=np.complex64)    # note: this cast is unnecessary, as we actually don't get complex values
        for i in range(0, nParts):
            y_hat[:,i] = self.qftInst.transform(y_split_list[i], suppressPrint=True)

        return y_hat