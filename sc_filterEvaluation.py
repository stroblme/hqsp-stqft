from math import exp
from qft import qft_framework, loadBackend, get_fft_from_counts, hexKeyToBin
from fft import fft_framework
from frontend import frontend, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "filterEvaluation"
# export.checkWorkingTree()

nQubits = 4
b = 1
mrot = PI/2**(nQubits-b-1)

print(f"Mrot set to {mrot}")

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nQubits)

y.addFrequency(125)  # you should choose a frequency matching a multiple of samplingRate/nSamples
y.addFrequency(250)

plotData = y.show(subplot=[1,6,1], title='signal')

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, f"Flat, zero-like signal, 2^{nQubits} samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing FFT")

fft = transform(fft_framework)

y_hat, f = fft.forward(y)
y_hat_ideal_p, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal_p, f_p, subplot=[1,6,2], title="FFT (ref)")

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal_p)
exp.setData(export.DESCRIPTION, "FFT output")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

device = "ibmq_casablanca"
_, backend = loadBackend(simulation=False, backendName=device)

print("Processing Simulated QFT")

qft = transform(qft_framework, minRotation=mrot, numOfShots=4096, suppressPrint=False)

y_hat, f = qft.forward(y)
y_hat_sim_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_sim_p, f_p, subplot=[1,6,3], title=f"qft_sim")

exp = export(topic=TOPIC, identifier="qft_sim")
exp.setData(export.SIGNAL, y_hat_sim_p)
exp.setData(export.DESCRIPTION, "QFT, simulated, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, qft.transformation.getBackend())
exp.doExport()

print("Processing simulated qft with noise")

qft = transform(qft_framework, suppressPrint=False, numOfShots=4096, minRotation=mrot, suppressNoise=True, simulation=True, backendName=device)

y_hat_f, f = qft.forward(y)
y_hat_sim_n_p_f, f_p = qft.postProcess(y_hat_f, f)
plotData = qft.show(y_hat_sim_n_p_f, f_p, subplot=[1,6,4], title=f"qft_sim_n_f")

exp = export(topic=TOPIC, identifier="qft_sim_n_f")
exp.setData(export.SIGNAL, y_hat_sim_n_p_f)
exp.setData(export.DESCRIPTION, f"QFT, simulated, noise from {device}, filtered, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, qft.transformation.getBackend())
exp.setData(export.JOBRESULT, qft.transformation.lastJobResultCounts)
exp.setData(export.FILTERRESULT, qft.transformation.filterResultCounts)
exp.doExport()

import numpy as np
y_hat  = np.array(get_fft_from_counts(qft.transformation.lastJobResultCounts, nQubits))
y_hat_sim_n_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_sim_n_p, f_p, subplot=[1,6,5], title=f"qft_sim_n")

exp = export(topic=TOPIC, identifier="qft_sim_n")
exp.setData(export.SIGNAL, y_hat_sim_n_p)
exp.setData(export.DESCRIPTION, f"QFT, simulated, noise from {device}, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, qft.transformation.getBackend())
exp.setData(export.JOBRESULT, qft.transformation.lastJobResultCounts)
exp.setData(export.FILTERRESULT, qft.transformation.filterResultCounts)
exp.doExport()

y_hat_f  = np.array(get_fft_from_counts(*hexKeyToBin(qft.transformation.filterResultCounts, nQubits)))
y_hat_sim_n_p_f, f_p = qft.postProcess(y_hat_f, f)
plotData = qft.show(y_hat_sim_n_p_f, f_p, subplot=[1,6,6], title=f"qft_filter")

# print("Processing real qft")

# qft = transform(qft_framework, fixZeroSignal=True, suppressPrint=False, suppressNoise=True, simulation=False, backendName=device, filterBackend=qft.transformation.getBackend())

# y_hat, f = qft.forward(y)
# y_hat_real_p, f_p = qft.postProcess(y_hat, f)
# plotData = qft.show(y_hat_real_p, f_p, subplot=[1,6,5], title=f"qft_real")

# exp = export(topic=TOPIC, identifier="qft_real")
# exp.setData(export.SIGNAL, y_hat_real_p)
# exp.setData(export.DESCRIPTION, f"QFT, real Device {device}, zeroSignalFix")
# exp.setData(export.PLOTDATA, plotData)
# exp.setData(export.BACKEND, qft.transformation.getBackend())
# exp.doExport()

print("Showing all figures")
frontend.primeTime() # Show all with blocking