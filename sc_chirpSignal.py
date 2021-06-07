from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stqft import stqft_framework
from frontend import signal, transform, primeTime, enableInteractive, setStylesheet

enableInteractive()
setStylesheet('dark_background') #seaborn-poster, seaborn-deep

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**10, signalType='chirp')

y.addFrequency(100)
y.addFrequency(500, 0.01)

y.show(subplot=[1,3,1])
print("Processing DFT")


print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
fft.show(y_hat, f, subplot=[1,3,2])


print("Processing STQFT")
stqft = transform(stqft_framework)
y_hat, f = stqft.forward(y, 2**4)
stqft.show(y_hat, f, subplot=[1,3,3])


print("Showing all figures")
primeTime() # Show all with blocking