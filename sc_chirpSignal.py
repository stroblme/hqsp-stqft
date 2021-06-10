from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

windowLength = 2**7
overlapFactor=0.5
windowType='hanning'

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

y.show(subplot=[1,3,1])



print("Processing STFT")
stft = transform(stft_framework)
y_hat, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
stft.show(y_hat, f, t, subplot=[1,3,2])


print("Processing STQFT")
stqft = transform(stqft_framework, suppressPrint=True)
y_hat, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
stqft.show(y_hat, f, t, subplot=[1,3,3])


print("Showing all figures")
primeTime() # Show all with blocking