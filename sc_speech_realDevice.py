from matplotlib.pyplot import draw
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "speech_realDevice"
export.checkWorkingTree()

print("Initializing Harmonic Signal")

y = signal(samplingRate=16000, amplification=1, duration=1, nSamples=2**16)

y.addFrequency(800)
y.addFrequency(2000)

plotData = y.show(subplot=[2,2,1])

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Harmonic Signal, 125 and 250 Hz at 1kHz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing QFT")

qft = transform(qft_framework, minRotation=0.2, suppressPrint=False)
y_hat, f = qft.forward(y)
y_hat_p, f_p = qft.postProcess(y_hat, f)
plotData = qft.show(y_hat_p, f_p, subplot=[2,2,2])

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal)
exp.setData(export.DESCRIPTION, "FFT, default param, post processed")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

nQubits = 7
bandwidth = 4
windowLength = 2**nQubits
overlapFactor=0.5
windowType='hann'

print("Initializing Chirp")

del(y)

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

y.show(subplot=[2,2,3])

print("Processing STQFT")
stqft = transform(stqft_framework, minRotation=PI/2**(nQubits-bandwidth), suppressPrint=False, draw=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[2,2,4])



print("Showing all figures")
frontend.primeTime() # Show all with blocking