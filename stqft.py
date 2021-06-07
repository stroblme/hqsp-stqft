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


    def transform(self, y_signal, nParts, show=-1):
        """Apply QFT on a given Signal

        Args:
            y (signal): signal to be transformed

        Returns:
            signal: transformeed signal
        """
        y_split_list = y_signal.split(nParts)

        y_hat = np.array([])
        for y_split in y_split_list:
            y_hat_split = self.qftInst.transform(y_split, suppressPrint=True)

            # qft.show(y_hat, f, subplot=[2,2,4])
            y_hat = np.append(y_hat, y_hat_split)

        # print(f"Stretching signal with scalar {self.scaler}")
        # y_preprocessed = self.preprocessSignal(y, self.scaler)

        # x_processed = x_processed[2:4]
        # print(f"Calculating required qubits for encoding a max value of {int(max(y_preprocessed))}")
        # circuit_size = int(max(y_preprocessed)).bit_length() # this basically defines the "adc resolution"

        # y_hat = self.processQFT_dumb(y_preprocessed, circuit_size, show)
        # y_hat = self.processQFT_layerwise(y_preprocessed, circuit_size, show)
        # y_hat = self.processQFT_geometric(y_preprocessed, circuit_size, show)
        return y_hat

