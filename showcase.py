from qft import qft_framework
from dft import dft_framework
from frontend import signal, transform, primeTime

print("Initializing Signal")

y = signal(samplingRate=44100, amplification=1, duration=10, nSamples=65536)

y.addFrequency(600)
y.addFrequency(800)
y.addFrequency(1000)
y.addFrequency(2000)
y.addFrequency(5000)



y.sample()
y.show()

# print("Processing DFT")

# dft = transform(dft_framework)
# y_hat, f = dft.forward(y)
# dft.show(y_hat, f)

print("Processing QFT")

qft = transform(qft_framework)

# qft.transformation.showCircuit(y.sample())

y_hat, f = qft.forward(y)

import numpy as np

y_hat = np.array(y_hat)

qft.show(y_hat, f, isOneSided=True)


primeTime() # Show all with blocking