import scipy
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

t = np.arange(400)

n = np.zeros((400,), dtype=complex)

n[40:60] = np.sin(1j*rng.uniform(0, 2*np.pi, (20,)))

s = scipy.fft.fft(n)

plt.plot(t, s.real, 'b-', t, s.imag, 'r--')

plt.legend(('real', 'imaginary'))

plt.show()