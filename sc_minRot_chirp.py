from math import exp
from matplotlib.pyplot import draw, text
from numpy.random import random
from qiskit.providers import backend
from qft import qft_framework
from dft import dft_framework
from fft import fft_framework
from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, grader, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "minRot_chirp"
export.checkWorkingTree()

windowLength = 2**7
overlapFactor=0.5
windowType='hann'

print("Initializing Chirp Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

plotData = y.show(subplot=[2,9,1])

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Chirp Signal, 500 to 2000 Hz and 1000 to 3000 Hz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing STFT")

stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,4,2])

exp = export(topic=TOPIC, identifier="stft")
exp.setData(export.SIGNAL, y_hat_stft)
exp.setData(export.DESCRIPTION, "STFT, default param, post processed")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Simulated STQFT")

mrot = 0
pt = 0
grader_inst = grader()

while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False)
    stqft = transform(stqft_framework, minRotation=0.2, suppressPrint=True, backend="simu")
    y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_sqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel')
    plotData = stqft.show(y_hat_sqft_p, f_p, t_p, subplot=[1,4,3])
    
    snr = grader_inst.calculateNoisePower(y_hat_sqft_p, y_hat_stft)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

    exp = export(topic=TOPIC, identifier=f"qft_sim_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_sqft_p)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, mrot={mrot}, post processed")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

plotData = grader_inst.show(subplot=[2,9,9])

exp = export(topic=TOPIC, identifier="grader_qft_sim")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, qft_sim")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Real QFT")

mrot = 0
pt = 0
grader_inst = grader()
import random
while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False, simulation=True, backendName="ibmq_quito")
    y_hat, f = qft.forward(y)
    y_hat_real, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = qft.show(y_hat_real, f_p, subplot=[2,9,pt+12], title=f"QFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)
    snr = grader_inst.calculateNoisePower(y_hat_real, y_hat_stft)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")
    pt += 1
    mrot = PI/2**(5-pt)

    exp = export(topic=TOPIC, identifier=f"qft_real_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_real)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, ibmq_quito noise, mrot={mrot}, post processed")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

plotData = grader_inst.show(subplot=[2,9,18])

exp = export(topic=TOPIC, identifier="grader_qft_real")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, qft_real")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

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
frontend.primeTime() # Show all with blocking