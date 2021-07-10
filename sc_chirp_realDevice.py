from matplotlib.pyplot import draw, text
from numpy.random import random
from qiskit.providers import backend
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, grader, signal, transform, export

from utils import PI

frontend.enableInteractive()

windowLength = 2**4
overlapFactor=0.5
windowType='hann'

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

y.show(subplot=[1,4,1], title="signal")

print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,4,2], title="stft")

print("Processing simulation STQFT")
stqft = transform(stqft_framework, minRotation=0.2, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,4,3], title="stqft")

# print("Processing real STQFT")
# stqft = transform(stqft_framework, minRotation=0, suppressPrint=True, simulation=True, backendName="ibmq_casablanca")
# y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
# stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,4,4])

print("Showing all figures")
frontend.primeTime() # Show all with blocking