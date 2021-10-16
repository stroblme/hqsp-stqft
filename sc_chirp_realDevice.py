from qiskit.providers import backend
from stft import stft_framework
from stqft import stqft_framework
from qft import setupMeasurementFitter, loadBackend, loadNoiseModel
from frontend import frontend, grader, signal, transform, export

from utils import PI

frontend.enableInteractive()
# TOPIC = "chirp_realDevice_guadalupe"
# TOPIC = "chirp_realDevice_casablanca"
# export.checkWorkingTree()
# device = "ibmq_guadalupe"

nQubits=10
samplingRate=16000    #careful: this may be modified when calling gen_features
numOfShots=4096
signalThreshold=0.06 #optimized according to thesis
minRotation=0.2 #PI/2**(nQubits-4)
nSamplesWindow=1024
overlapFactor=0.875
windowLength = 2**nQubits
windowType='blackman'
suppressPrint=True
useNoiseModel=True
backend="ibmq_guadalupe" #ibmq_guadalupe, ibmq_melbourne (noisier)
noiseMitigationOpt=0
numOfRuns=1
simulation=True
transpileOnce=True
transpOptLvl=1
fixZeroSignal=False
scale='mel'
normalize=True
nMels=60
fmin=40.0
enableQuanv=True

# mrot = PI/2**(nQubits-1-b)
mrot = 0
print(f"Mrot set to {mrot}")

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

# _, backendInstance = loadBackend(backendName=backend, simulation=True)
# _, noiseModel = loadNoiseModel(backendName=backendInstance)
# filterResultCounts = setupMeasurementFitter(backendInstance, noiseModel,
#                                                     transpOptLvl=1, nQubits=nQubits, nShots=nShots)
filterResultCounts = None

# print("Processing simulation STQFT with noise, mitigated")
# stft = transform(stft_framework)
# y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
# y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
# plotData = stft.show(y_hat_stft_p, f_p, t_p, title="stft")

# QFT init
stqft = transform(stqft_framework, 
                    numOfShots=numOfShots, 
                    minRotation=minRotation, fixZeroSignal=fixZeroSignal,
                    suppressPrint=suppressPrint, draw=False,
                    simulation=simulation,
                    transpileOnce=transpileOnce, transpOptLvl=transpOptLvl)

# STQFT init
y_hat_stqft, f, t = stqft.forward(y, 
                        nSamplesWindow=nSamplesWindow,
                        overlapFactor=overlapFactor,
                        windowType=windowType,
                        suppressPrint=suppressPrint)

y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=None, normalize=normalize, samplingRate=y.samplingRate, nMels=nMels, fmin=fmin, fmax=y.samplingRate/2)

stqft.show(y_hat_stqft_p, f_p, t_p, title=f"STQFT")


print("Showing all figures")
frontend.primeTime() # Show all with blocking