from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet

from tests import *

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

speechSignal = '../dataset/zero/0c40e715_nohash_1.wav'

windowLength = 2**10
overlapFactor=0.5
windowType='hanning'

print("Initializing Signal")

# y = signal(samplingRate=16000, signalType='file', path=speechSignal)
windowLength = 2**10
y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**16, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)
y.show(subplot=[1,3,1])



print("Processing STFT")
stft = transform(stft_framework)
y_hat, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
stft.show(y_hat, f, t=t, subplot=[1,3,2], scale='mel', fmax=8000)

y_hat, f, t = test_stft(y)
stft.show(y_hat, f, t=t, subplot=[1,3,3], scale='mel', fmax=8000)


# # test_plot(test_to_db(test_mel(y_hat, y.samplingRate)), y.samplingRate)
# test_plot(y_hat, y.samplingRate)

# # print("Processing STQFT")
# # stqft = transform(stqft_framework)
# # y_hat, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# # stqft.show(y_hat, f, t=t, subplot=[1,3,3], scale='mel', fmax=8000)

# print("Running reference")
# test_melspectogram(y)

print("Showing all figures")
primeTime() # Show all with blocking