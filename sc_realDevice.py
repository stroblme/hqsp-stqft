from matplotlib.pyplot import draw
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive

enableInteractive()

print("Initializing Harmonic Signal")

y = signal(samplingRate=16000, amplification=1, duration=1, nSamples=2**16)

y.addFrequency(800)
y.addFrequency(2000)

y.show(subplot=[2,2,1])

print("Processing QFT")

qft = transform(qft_framework, minRotation=0.2, suppressPrint=False)
y_hat, f = qft.forward(y)
y_hat_p, f_p = qft.postProcess(y_hat, f)
qft.show(y_hat_p, f_p, subplot=[2,2,2])

windowLength = 2**5 #only 5 qubits allowed
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
stqft = transform(stqft_framework, minRotation=0.2, suppressPrint=False, simulation=False)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[2,2,4])



print("Showing all figures")
primeTime() # Show all with blocking