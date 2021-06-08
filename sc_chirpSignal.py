from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(900)
y.addFrequency(1050, 0.008)

y.addFrequency(1900)
y.addFrequency(2050, 0.008)

y.show(subplot=[1,3,1])

windowLength = 2**7

print("Processing STFT")
stft = transform(stft_framework)
y_hat, f ,t = stft.forward(y, windowLength)
stft.show(y_hat, f, t, subplot=[1,3,2])


print("Processing STQFT")
stqft = transform(stqft_framework)
y_hat, f, t = stqft.forward(y, windowLength)
stqft.show(y_hat, f, t, subplot=[1,3,3])


print("Showing all figures")
primeTime() # Show all with blocking