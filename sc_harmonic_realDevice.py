from math import exp
from matplotlib.pyplot import draw, text
from numpy.random import random
from qiskit.providers import backend
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import grader, signal, transform, primeTime, enableInteractive, export
from utils import PI

enableInteractive()
TOPIC = "harmonic_realDevice"
export.checkWorkingTree()

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**4)

y.addFrequency(250) # you should choose a frequency matching a multiple of samplingRate/nSamples
y.addFrequency(125) # you should choose a frequency matching a multiple of samplingRate/nSamples

plotData = y.show(subplot=[2,9,1])

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Harmonic Signal, 125 and 250 Hz at 1kHz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
y_hat_ideal, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal, f_p, subplot=[2,9,10], title="FFT (ref)")

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal)
exp.setData(export.DESCRIPTION, "FFT, default param, post processed")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Simulated QFT")

mrot = 0
pt = 0
grader_inst = grader()

while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False)
    y_hat, f = qft.forward(y)
    y_hat_real, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = qft.show(y_hat_real, f_p, subplot=[2,9,pt+3], title=f"QFT_sim, mr:{mrot:.2f}", xlabel=" ", ylabel=ylabel)
    snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_ideal)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

    exp = export(topic=TOPIC, identifier=f"qft_sim_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_real)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, mrot={mrot}, post processed")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

plotData = grader_inst.show(subplot=[2,9,9])

exp = export(topic=TOPIC, identifier="grader_qft_sim")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, qft_sim")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Real QFT")

mrot = 0
pt = 0
grader_inst = grader()
import random
while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False, simulation=True, backendName="ibmq_quito")
    y_hat, f = qft.forward(y)
    y_hat_real, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = qft.show(y_hat_real, f_p, subplot=[2,9,pt+12], title=f"QFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)
    snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_ideal)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

    exp = export(topic=TOPIC, identifier=f"qft_real_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_real)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, ibmq_quito noise, mrot={mrot}, post processed")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

plotData = grader_inst.show(subplot=[2,9,18])

exp = export(topic=TOPIC, identifier="grader_qft_real")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, qft_real")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

# grader_inst = grader()
# snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_sim)
# print(f"Calculated an snr of {snr} db")
# windowLength = 2**5 #only 5 qubits allowed
# overlapFactor=0.5
# windowType='hann'

# print("Initializing Chirp")

# del(y)

# y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

# y.addFrequency(500)
# y.addFrequency(2000, y.duration)

# y.addFrequency(1000)
# y.addFrequency(3000, y.duration)

# y.show(subplot=[2,2,3])

# print("Processing STQFT")
# stqft = transform(stqft_framework, minRotation=0.2, suppressPrint=False, simulation=False)
# y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
# stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[2,2,4])



print("Showing all figures")
primeTime() # Show all with blocking