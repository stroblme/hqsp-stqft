from qiskit.providers import backend
from stft import stft_framework
from stqft import stqft_framework
from qft import setupMeasurementFitter, loadBackend, loadNoiseModel
from frontend import frontend, grader, signal, transform, export

from utils import PI

frontend.enableInteractive()
TOPIC = "chirp_realDevice_mrot_mitig"
# TOPIC = "chirp_realDevice_mrot"
export.checkWorkingTree()
device = "ibmq_casablanca" #noisy
# device = "ibmq_guadalupe"

nQubits = 7
windowLength = 2**nQubits
overlapFactor=0.5
windowType='hanning'
b = 2
w=4
# mrot = PI/2**(nQubits-1)
mrot=0
# mrot = PI/2**(nQubits-1-b)

print(f"Mrot set to {mrot}")

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

nShots=2048

_, backendInstance = loadBackend(backendName=device, simulation=True)
_, noiseModel = loadNoiseModel(backendName=backendInstance)
filterResultCounts = setupMeasurementFitter(backendInstance, noiseModel,
                                                    transpOptLvl=1, nQubits=nQubits, nShots=nShots)

mrot = 0
pt = 0

while mrot <= PI/2:

    print("Processing simulation STQFT with noise, mitigated")
    stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, suppressPrint=True, noiseModel=noiseModel, backend=backendInstance, noiseMitigationOpt=1,  filterResultCounts=filterResultCounts, numOfShots=nShots)
    
    # stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, noiseModel=noiseModel, backend=backendInstance, suppressPrint=True)

    ylabel = "" if pt == 0 else " "


    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
    plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[2,nQubits/2+1,pt+1], title=f"STQFT_real, mr:{mrot:.2f}", ylabel=ylabel)

    exp = export(topic=TOPIC, identifier=f"stqft_real_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_stqft_p)
    exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
    exp.setData(export.PLOTDATA, plotData)
    exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
    exp.doExport()

    pt += 1
    mrot = PI/2**(nQubits-pt)

print("Showing all figures")
frontend.primeTime() # Show all with blocking