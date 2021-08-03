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
mrot = 0.2

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


stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True)
y_hat_sim, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sim_p, f_p, t_p = stqft.postProcess(y_hat_sim, f ,t, scale='mel')
plotData = stqft.show(y_hat_sim_p, f_p, t_p, subplot=[2,nQubits+3,1], title=f"STQFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel="Amplitude")

print("Processing Inverse STFT")

y_i = y

y_hat = signal()
y_hat.externalSample(y_hat_sim_p, t)

stft = transform(stft_framework, transform=ifft_framework)
y_i_samples, f ,t = stft.forward(y_hat, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_i.externalSample(y_i_samples,t)
y_i.show(subplot=[2,nQubits+3,2])

print("Showing all figures")
frontend.primeTime() # Show all with blocking