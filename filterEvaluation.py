from math import exp
from qft import qft_framework
from fft import fft_framework
from frontend import frontend, signal, transform, export


frontend.enableInteractive()
TOPIC = "filterEvaluation"
export.checkWorkingTree()

device = "ibmq_quito"

nQubits = 4

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nQubits)

y.addFrequency(125)
y.addFrequency(250)

plotData = y.show(subplot=[1,5,1], title='signal')

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, f"Flat, zero-like signal, 2^{nQubits} samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing FFT")

fft = transform(fft_framework)

y_hat, f = fft.forward(y)
y_hat_ideal_p, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal_p, f_p, subplot=[1,5,2], title="FFT (ref)")

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal_p)
exp.setData(export.DESCRIPTION, "FFT output")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Simulated QFT")

qft = transform(qft_framework, fixZeroSignal=True, suppressPrint=False, simulation=True, backendName=device)

y_hat, f = qft.forward(y)
y_hat_sim_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_sim_p, f_p, subplot=[1,5,3], title=f"qft_sim")

exp = export(topic=TOPIC, identifier="qft_sim")
exp.setData(export.SIGNAL, y_hat_sim_p)
exp.setData(export.DESCRIPTION, "QFT, simulated, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing simulated qft with noise")

qft = transform(qft_framework, fixZeroSignal=True, suppressPrint=False, suppressNoise=True, simulation=True, backendName=device)

y_hat, f = qft.forward(y)
y_hat_sim_n_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_sim_n_p, f_p, subplot=[1,5,4], title=f"qft_sim_n")

exp = export(topic=TOPIC, identifier="qft_sim_n")
exp.setData(export.SIGNAL, y_hat_sim_n_p)
exp.setData(export.DESCRIPTION, f"QFT, simulated, noise from {device}, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing real qft")

qft = transform(qft_framework, fixZeroSignal=True, suppressPrint=False, suppressNoise=True, simulation=False, backendName=device, filterBackend=qft.transformation.getBackend())

y_hat, f = qft.forward(y)
y_hat_real_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_real_p, f_p, subplot=[1,5,5], title=f"qft_real")

exp = export(topic=TOPIC, identifier="qft_real")
exp.setData(export.SIGNAL, y_hat_real_p)
exp.setData(export.DESCRIPTION, f"QFT, real Device {device}, zeroSignalFix")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Showing all figures")
frontend.primeTime() # Show all with blocking