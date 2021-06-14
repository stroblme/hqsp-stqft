from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet, grader

from tests import *

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

speechSignal = '../dataset/zero/0c40e715_nohash_1.wav'

windowLength = 2**10
overlapFactor=0.5
windowType='hann'

print("Initializing Signal")

y = signal(samplingRate=16000, signalType='file', path=speechSignal)
y.show(subplot=[1,4,1])

print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t, scale='mel', fmax=4000)
stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,4,2])


print("Processing STQFT")
stqft = transform(stqft_framework, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stft.postProcess(y_hat_stqft, f ,t, scale='mel', fmax=4000)
stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,4,3])

grader_inst = grader()
y_hat_diff = grader_inst.correlate2d(y_hat_stft_p, y_hat_stqft_p)
grader_inst.show(y_hat_diff, f_p, t=t_p, subplot=[1,4,4])


print("Showing all figures")
primeTime() # Show all with blocking