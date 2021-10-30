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

nQubits=7
samplingRate=8000    #careful: this may be modified when calling gen_features
numOfShots=4096
signalThreshold=0.06 #optimized according to thesis
minRotation=0.2 #PI/2**(nQubits-4)
overlapFactor=0.5
windowLength = 2**nQubits
windowType='hanning'
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

y = signal(samplingRate=8000, amplification=1, duration=4.1, signalType='chirp')

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
# # plotData = stft.show(y_hat_stft_p, f_p, t_p, title="stft")

# stqft = transform(stqft_framework, 
#                         numOfShots=numOfShots, 
#                         minRotation=minRotation, fixZeroSignal=fixZeroSignal,
#                         suppressPrint=suppressPrint, draw=False,
#                         simulation=simulation,
#                         transpileOnce=transpileOnce, transpOptLvl=transpOptLvl)

# # STQFT init
# y_hat_stqft, f, t = stqft.forward(y, 
#                         nSamplesWindow=windowLength,
#                         overlapFactor=overlapFactor,
#                         windowType=windowType,
#                         suppressPrint=suppressPrint)


# y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=None, normalize=normalize, samplingRate=y.samplingRate)

# stqft.show(y_hat_stqft_p, f_p, t_p)

# QFT init
for i in range(1,5):

    stqft = transform(stqft_framework, 
                        numOfShots=numOfShots, 
                        minRotation=minRotation, fixZeroSignal=fixZeroSignal,
                        suppressPrint=suppressPrint, draw=False,
                        simulation=simulation,
                        transpileOnce=transpileOnce, transpOptLvl=transpOptLvl)

    # STQFT init
    y_hat_stqft, f, t = stqft.forward(y, 
                            nSamplesWindow=windowLength,
                            overlapFactor=overlapFactor,
                            windowType=windowType,
                            suppressPrint=suppressPrint)

    ylabel = "Frequency (Hz)" if i == 1 else " "
    xlabel = "Time (s)"

    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=None, normalize=normalize, samplingRate=y.samplingRate)

    stqft.show(y_hat_stqft_p, f_p, t_p, title=f"STQFT_sim, er:{0.12+0.03*(i-1)}", subplot=[1,4,i], xlabel=xlabel, ylabel=ylabel)


print("Showing all figures")
frontend.primeTime() # Show all with blocking