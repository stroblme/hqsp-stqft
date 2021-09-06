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

class qwvt_framework():
    def __init__(self, **kwargs):
        self.stt_inst = stt_framework(qft_framework, **kwargs)

    def transform(self, y_signal, **kwargs):
        return self.stt_inst.stt_transform(y_signal, **kwargs)

    def wvd(y_signal, t=None, N=None, trace=0, make_analytic=True):
        y=y_signal.sample()
        # if make_analytic:
        #     x = hilbert(y)
        # else:
        x = array(y)

        if x.ndim == 1: [xrow, xcol] = np.shape(array([x]))
        else: raise ValueError("Signal x must be one-dimensional.")

        if t is None: t = arange(len(x))
        if N is None: N = len(x)

        if (N <= 0 ): raise ValueError("Number of Frequency bins N must be greater than zero.")

        if t.ndim == 1: [trow, tcol] = np.shape(array([t]))
        else: raise ValueError("Time indices t must be one-dimensional.")


        tfr = zeros([N, tcol], dtype='complex')
        if trace: print "Wigner-Ville distribution",
        for icol in xrange(0, tcol):
            ti = t[icol]
            taumax = min([ti, xcol-ti-1, int(round(N/2.0))-1])
            tau = arange(-taumax, taumax+1)
            indices = ((N+tau)%N)
            tfr[ix_(indices, [icol])] = transpose(array(x[ti+tau] * conj(x[ti-tau]), ndmin=2))
            tau=int(round(N/2))+1
            if ((ti+1) <= (xcol-tau)) and ((ti+1) >= (tau+1)):
                if(tau >= tfr.np.shape[0]): tfr = append(tfr, zeros([1, tcol]), axis=0)
                tfr[ix_([tau], [icol])] = array(0.5 * (x[ti+tau] * conj(x[ti-tau]) + x[ti-tau] * conj(x[ti+tau])))
            if trace: disprog(icol, tcol, 10)

        tfr = real(fft.fft(tfr, axis=0))
        f = 0.5*arange(N)/float(N)
        return (transpose(tfr), t, f )


