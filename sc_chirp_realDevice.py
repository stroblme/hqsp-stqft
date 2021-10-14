from qiskit.providers import backend
from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, grader, signal, transform, export

from utils import PI

frontend.enableInteractive()
TOPIC = "chirp_realDevice"
export.checkWorkingTree()
device = "ibmq_casablanca"

nQubits = 7
windowLength = 2**nQubits
overlapFactor=0.5
windowType='hanning'
b = 2
w=4
mrot = PI/2**(nQubits-1-b)
print(f"Mrot set to {mrot}")

print("Initializing Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

# plotData = y.show(subplot=[1,w,1], title="signal")

# exp = export(topic=TOPIC, identifier="signal")
# # exp.setData(export.SIGNAL, y)
# exp.setData(export.DESCRIPTION, "Harmonic Signal, 125 and 250 Hz at 1kHz, 2^4 samples")
# exp.setData(export.PLOTDATA, plotData)
# exp.doExport()

print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,w,1], title="stft")
# plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,w,2], title="stft")

exp = export(topic=TOPIC, identifier="stft")
exp.setData(export.SIGNAL, y_hat_stft)
exp.setData(export.DESCRIPTION, "stft, chirp, window: 'hanning', length=2**7")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing simulation STQFT")
stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,w,2], title="stqft")
# plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,w,3], title="stqft")

exp = export(topic=TOPIC, identifier="stqft")
exp.setData(export.SIGNAL, y_hat_stqft)
exp.setData(export.DESCRIPTION, f"stqft, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing simulation STQFT with noise")
stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, backend=device)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,w,3], title="stqft_noise")
# plotData = stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,w,4], title="stqft_noise")

exp = export(topic=TOPIC, identifier="stqft-noise")
exp.setData(export.SIGNAL, y_hat_stqft)
exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
exp.doExport()

print("Processing simulation STQFT with noise, mitigated")
stqft = transform(stqft_framework, minRotation=mrot, simulation=True, useNoiseModel=True, backend=device, noiseMitigationOpt=1)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t)
plotData = stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,w,4], title="stqft_noise")
# plotData = stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,w,4], title="stqft_noise")

exp = export(topic=TOPIC, identifier="stqft-noise-mitigated")
exp.setData(export.SIGNAL, y_hat_stqft)
exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{mrot}")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
exp.doExport()



print("Showing all figures")
frontend.primeTime() # Show all with blocking