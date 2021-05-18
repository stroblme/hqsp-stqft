from qft import qft_framework
from dft import dft_framework
from frontend import signal, transform, primeTime

print("Initializing Signal")

y = signal()
y.addFrequency(3., 0.2)
y.addFrequency(5.)
y.addFrequency(7.)


y.sample()
y.show()

print("Processing DFT")

dft = transform(dft_framework)
y_hat, f = dft.forward(y)
dft.show(y_hat, f)

print("Processing QFT")

qft = transform(qft_framework)

# qft.transformation.showCircuit(y.sample())

y_hat, f = qft.forward(y)
qft.show(y_hat, f)


primeTime() # Show all with blocking