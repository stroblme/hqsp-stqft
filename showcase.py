from qft import qft_framework
from dft import dft_framework
from frontend import signal, transform, primeTime

y = signal()
y.addFrequency(2.)

y.sample()
y.show()


dft = transform(dft_framework)
y_hat, f = dft.forward(y)
dft.show(y_hat, f)


qft = transform(qft_framework)
y_hat, f = qft.forward(y)
qft.show(y_hat, f)


primeTime() # Show all with blocking