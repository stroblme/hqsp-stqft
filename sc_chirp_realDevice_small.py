from qiskit.providers import backend
from stft import stft_framework
from stqft import stqft_framework
from qft import setupMeasurementFitter, loadBackend, loadNoiseModel
from frontend import frontend, grader, signal, transform, export

from utils import PI

frontend.enableInteractive()
TOPIC = "chirp_realDevice_guadalupe"
# TOPIC = "chirp_realDevice_casablanca"
export.checkWorkingTree()
device = "ibmq_guadalupe"
# device = "ibmq_casablanca"

nQubits = 7
windowLength = 2**nQubits
overlapFactor=0.5
windowType='hanning'
b = 2
w=4
nShots=2048

# mrot = PI/2**(nQubits-1-b)
mrot = 0
print(f"Mrot set to {mrot}")

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)


print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,w,1], title="stft")

exp = export(topic=TOPIC, identifier="stft")
exp.setData(export.SIGNAL, y_hat_stft_p)
exp.setData(export.DESCRIPTION, "stft, chirp, window: 'hanning', length=2**7")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()






print("Processing simulation STQFT")
stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,w,2], title="STQFT_sim", ylabel=" ")

exp = export(topic=TOPIC, identifier="stqft")
exp.setData(export.SIGNAL, y_hat_stqft_p)
exp.setData(export.DESCRIPTION, f"stqft, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()


_, backendInstance = loadBackend(backendName=device, simulation=True)
_, noiseModel = loadNoiseModel(backendName=backendInstance)

print("Processing simulation STQFT with noise")
stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, noiseModel=noiseModel, backend=backendInstance, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,w,3], title="STQFT_sim_n", ylabel=" ")

exp = export(topic=TOPIC, identifier="stqft-noise")
exp.setData(export.SIGNAL, y_hat_stqft_p)
exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
exp.doExport()




_, backendInstance = loadBackend(backendName=device, simulation=True)
_, noiseModel = loadNoiseModel(backendName=backendInstance)
filterResultCounts = setupMeasurementFitter(backendInstance, noiseModel,
                                                    transpOptLvl=1, nQubits=nQubits, nShots=nShots)

print("Processing simulation STQFT with noise, mitigated")
stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, suppressPrint=True, noiseModel=noiseModel, backend=backendInstance, noiseMitigationOpt=1,  filterResultCounts=filterResultCounts, numOfShots=nShots)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,w,4], title="STQFT_sim_n, mitigated", ylabel=" ")

exp = export(topic=TOPIC, identifier="stqft-noise-mitigated")
exp.setData(export.SIGNAL, y_hat_stqft_p)
exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
exp.doExport()



print("Showing all figures")
frontend.primeTime() # Show all with blocking