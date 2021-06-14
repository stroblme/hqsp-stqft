from IPython import get_ipython

import numpy as np
from numpy import array, pi

from math import log2

import matplotlib.pyplot as plt

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT as qiskit_qft

from qft import qft_framework
from stt import stt_framework

class stqft_framework():
    def __init__(self, **kwargs):
        self.stt_inst = stt_framework(qft_framework, **kwargs)

    def transform(self, y_signal, **kwargs):
        return self.stt_inst.stt_transform(y_signal, **kwargs)