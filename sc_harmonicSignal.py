from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from frontend import frontend, signal, transform

frontend.enableInteractive()

print("Initializing Signal")

y = signal(samplingRate=16000, amplification=1, duration=1, nSamples=2**16)

y.addFrequency(600)
y.addFrequency(800)
y.addFrequency(1000)
y.addFrequency(2000)
y.addFrequency(5000)

y.show()

print("Processing DFT")

try:
    dft = transform(dft_framework)
    y_hat, f = dft.forward(y)
    dft.show(y_hat, f, )
except Exception as e:
    print(e)

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
fft.show(y_hat, f)

print("Processing QFT")

qft = transform(qft_framework)
y_hat, f = qft.forward(y)
qft.show(y_hat, f)


print("Showing all figures")
frontend.primeTime() # Show all with blocking