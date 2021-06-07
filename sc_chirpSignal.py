from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from frontend import signal, transform, primeTime, enableInteractive

enableInteractive()

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=1, nSamples=2**4, signalType='chirp')

y.addFrequency(100)
y.addFrequency(500, 0.01)

y.show(subplot=[2,2,1])
print("Processing DFT")

try:
    dft = transform(dft_framework)
    y_hat, f = dft.forward(y)
    dft.show(y_hat, f, subplot=[2,2,2])
except Exception as e:
    print(e)

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
fft.show(y_hat, f, subplot=[2,2,3])

print("Processing QFT")

qft = transform(qft_framework, numOfShots=512)
y_hat, f = qft.forward(y)
qft.show(y_hat, f, subplot=[2,2,4])


print("Showing all figures")
primeTime(subplots=True) # Show all with blocking