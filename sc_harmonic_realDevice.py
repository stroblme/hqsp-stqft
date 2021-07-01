from matplotlib.pyplot import draw, text
from numpy.random import random
from qiskit.providers import backend
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import grader, signal, transform, primeTime, enableInteractive
from utils import PI

enableInteractive()

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**4)

y.addFrequency(250) # you should choose a frequency matching a multiple of samplingRate/nSamples
y.addFrequency(125) # you should choose a frequency matching a multiple of samplingRate/nSamples

y.show(subplot=[2,9,1])

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
y_hat_ideal, f_p = fft.postProcess(y_hat, f)
fft.show(y_hat_ideal, f_p, subplot=[2,9,10], title="FFT (ref)")


print("Processing Real QFT")

mrot = 0
pt = 0
grader_inst = grader()

while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False)
    y_hat, f = qft.forward(y)
    y_hat_real, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    qft.show(y_hat_real, f_p, subplot=[2,9,pt+3], title=f"QFT_sim, mr:{mrot:.2f}", xlabel=" ", ylabel=ylabel)
    snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_ideal)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

grader_inst.show(subplot=[2,9,9])


mrot = 0
pt = 0
grader_inst = grader()
import random
while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False, simulation=True, backendName="ibmq_quito")
    y_hat, f = qft.forward(y)
    y_hat_real, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    qft.show(y_hat_real, f_p, subplot=[2,9,pt+12], title=f"QFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)
    snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_ideal)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

grader_inst.show(subplot=[2,9,18])


# grader_inst = grader()
# snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_sim)
# print(f"Calculated an snr of {snr} db")
# windowLength = 2**5 #only 5 qubits allowed
# overlapFactor=0.5
# windowType='hann'

# print("Initializing Chirp")

# del(y)

# y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

# y.addFrequency(500)
# y.addFrequency(2000, y.duration)

# y.addFrequency(1000)
# y.addFrequency(3000, y.duration)

# y.show(subplot=[2,2,3])

# print("Processing STQFT")
# stqft = transform(stqft_framework, minRotation=0.2, suppressPrint=False, simulation=False)
# y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
# stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[2,2,4])



print("Showing all figures")
primeTime() # Show all with blocking