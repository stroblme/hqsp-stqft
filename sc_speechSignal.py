from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

speechSignal = '../dataset/zero/0c40e715_nohash_1.wav'

windowLength = 2**10
overlapFactor=0.5
windowType='hanning'

print("Initializing Signal")

y = signal(samplingRate=16000, signalType='file', path=speechSignal)

y.show(subplot=[1,3,1])



print("Processing STFT")
stft = transform(stft_framework)
y_hat, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
stft.show(y_hat, f, t, subplot=[1,3,2], scale='mel', fmax=4000)


print("Processing STQFT")
stqft = transform(stqft_framework)
y_hat, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
stqft.show(y_hat, f, t, subplot=[1,3,3], scale='mel', fmax=4000)


print("Showing all figures")
primeTime() # Show all with blocking