from qft import get_fft_from_counts


# --- Standard imports

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()

# --- Imports
from qiskit import QuantumCircuit, execute, BasicAer
from qiskit.tools.monitor import job_monitor

import math
from numpy import linalg as LA
import numpy as np
#%config jupy = 'svg' # Makes the images look nice

import time

import matplotlib.pyplot as plt


# --- Computation of the calibration matrix
nBits    = 4

shots    = 2048

from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
from qiskit import *
from qiskit.circuit.library import QFT

qr = QuantumRegister(nBits)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
backend = provider.get_backend('ibmq_quito')
job = execute(meas_calibs, backend=backend, shots=1000)
job_monitor(job, interval = 3)
cal_results = job.result()

meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
print(meas_fitter.cal_matrix)

# --- Execution of the noisy quantum circuit

q = QuantumRegister(4,'q')

qc = QuantumCircuit(q)
# Normalize ampl, which is required for squared sum of amps=1
ys = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nBits)

ys.addFrequency(125)
y = ys.sample()
# y.addFrequency(250)
ampls = y / np.linalg.norm(y)

# for 2^n amplitudes, we have n qubits for initialization
# this means that the binary representation happens exactly here

qc.initialize(ampls, [q[i] for i in range(nBits)])

qc += QFT(num_qubits=nBits, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')

qc.measure_all()
qc = transpile(qc, provider.get_backend('ibmq_quito'), optimization_level=1) # opt level 0,1..3. 3: heaviest opt
job = execute(qc, provider.get_backend('ibmq_quito'), shots = shots)
#job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots = shots)

job_monitor(job, interval = 3)
result = job.result()
print(result.get_counts())

# --- Error correction

# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(result)
mitigated_counts = mitigated_results.get_counts(0)

print(mitigated_counts)

y_hat = np.array(get_fft_from_counts(mitigated_counts, nBits))

print(y_hat)