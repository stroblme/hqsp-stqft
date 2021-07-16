from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, signal, transform
from tests import *

frontend.enableInteractive()

windowLength = 2**7
overlapFactor=0.5
windowType='hann'

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

y.show(subplot=[1,3,1])



print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,3,2])

# print("Running reference")
# y_hat_stft, f, t = test_stft_scipy(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
# stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,3,2])


print("Processing STQFT")
stqft = transform(stqft_framework, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,3,3])


print("Showing all figures")
frontend.primeTime() # Show all with blocking