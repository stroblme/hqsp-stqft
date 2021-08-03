from math import exp
from qft import loadBackend
from stft import stft_framework
from fft import ifft_framework
from stqft import stqft_framework
from frontend import frontend, grader, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "minRot_chirp"
export.checkWorkingTree()

nQubits = 7
windowLength = 2**nQubits #nqubits. using results from minRot_harmonic
overlapFactor=0.5
windowType='hann'

print("Initializing Chirp Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

plotData = y.show(subplot=[2,nQubits+2,1])

print("Processing STFT")

stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[2,nQubits+2,nQubits+3])

print("Processing Simulated STQFT")

mrot = 0
pt = 0
grader_inst = grader()

while mrot <= PI/2:
    stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True)
    y_hat_sim, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_sim_p, f_p, t_p = stqft.postProcess(y_hat_sim, f ,t, scale='mel')
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = stqft.show(y_hat_sim_p, f_p, t_p, subplot=[2,nQubits+2,2+pt], title=f"STQFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)

    snr = grader_inst.calculateNoisePower(y_hat_sim_p, y_hat_stft_p)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")

    pt += 1
    mrot = PI/2**(nQubits-pt)


print("Processing Inverse STFT")

stft = transform(stft_framework, ifft_framework)
y_hat_stft, f ,t = stft.backward(y_hat_sim_p, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[2,nQubits+2,nQubits+3])

print("Showing all figures")
frontend.primeTime() # Show all with blocking