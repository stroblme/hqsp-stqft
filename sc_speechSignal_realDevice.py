from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, signal, transform, export
from utils import PI
from tests import *

frontend.enableInteractive()
TOPIC = "speech_realDevice"
export.checkWorkingTree()

# speechSignal = '../dataset/zero/4a1e736b_nohash_2.wav' #male clear
# speechSignal = '../dataset/zero/0fa1e7a9_nohash_1.wav' #male noise
# speechSignal = '../dataset/zero/7ea032f3_nohash_3.wav'  #male questionary
# speechSignal = '../dataset/zero/8e05039f_nohash_4.wav'  #female clear
speechSignal = '../dataset/zero/4634529e_nohash_2.wav'  #female noise
# speechSignal = '/storage/mstrobl/dataset/zero/4634529e_nohash_2.wav'  #female noise


nQubits=10
samplingRate=16000    #careful: this may be modified when calling gen_features
numOfShots=4096
signalFilter=0.02
minRotation=PI/2**(nQubits-6)
nSamplesWindow=1024
overlapFactor=0.875
windowLength = 2**nQubits
windowType='blackman'
suppressPrint=True
scale='mel'
normalize=True
nMels=60
fmin=40.0

print("Initializing Signal")

y = signal(samplingRate=16000, signalType='file', path=speechSignal)
y.show(subplot=[1,3,1])

print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t, scale=scale, normalize=normalize, samplingRate=y.samplingRate, nMels=nMels, fmin=fmin, fmax=y.samplingRate/2)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,3,2], title="stft")

exp = export(topic=TOPIC, identifier="stft")
exp.setData(export.SIGNAL, y_hat_stft_p)
exp.setData(export.DESCRIPTION, "stft, chirp, window: 'hanning', length=2**7")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing STQFT")
# device = "ibmq_guadalupe"
# device = "ibmq_melbourne"
device = "ibmq_cambridge"

stqft = transform(stqft_framework, suppressPrint=False, signalFilter=signalFilter, minRotation=minRotation, simulation=True, useNoiseModel=True, backendName=device, transpileOnce=True)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale=scale, normalize=normalize, samplingRate=y.samplingRate, nMels=nMels, fmin=fmin, fmax=y.samplingRate/2)
plotData = stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,3,3], title="stqft_noise")

exp = export(topic=TOPIC, identifier="stqft-noise")
exp.setData(export.SIGNAL, y_hat_stqft_p)
exp.setData(export.DESCRIPTION, f"stqft, {device}, chirp, window: 'hanning', length=2**7, mrot:{minRotation}")
exp.setData(export.PLOTDATA, plotData)
exp.setData(export.BACKEND, stqft.transformation.stt_inst.transformationInst.getBackend())
exp.doExport()

# grader_inst = grader()
# y_hat_diff = grader_inst.correlate2d(y_hat_stft_p, y_hat_stqft_p)
# grader_inst.show(y_hat_diff, f_p, t=t_p, subplot=[1,4,4])


print("Showing all figures")
frontend.primeTime() # Show all with blocking