from math import exp
from qft import loadBackend
from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, grader, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "minRot_chirp"
export.checkWorkingTree()

nQubits = 7
windowLength = 2**nQubits #nqubits. using results from minRot_harmonic
overlapFactor=0.5
windowType='hanning'

print("Initializing Chirp Signal")

y = signal(samplingRate=8000, amplification=1, duration=0, nSamples=2**12, signalType='chirp')

y.addFrequency(500)
y.addFrequency(2000, y.duration)

y.addFrequency(1000)
y.addFrequency(3000, y.duration)

plotData = y.show(subplot=[2,nQubits+2,1])

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Chirp Signal, 500 to 2000 Hz and 1000 to 3000 Hz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing STFT")

stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t)
plotData = stft.show(y_hat_stft_p, f_p, t_p, subplot=[2,nQubits+2,nQubits+3])

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
    stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True)
    y_hat_sim, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_sim_p, f_p, t_p = stqft.postProcess(y_hat_sim, f ,t, scale='mel')
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = stqft.show(y_hat_sim_p, f_p, t_p, subplot=[2,nQubits+2,2+pt], title=f"STQFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)

    snr = grader_inst.calculateNoisePower(y_hat_sim_p, y_hat_stft_p)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")

    exp = export(topic=TOPIC, identifier=f"stqft_sim_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_sim_p)
    exp.setData(export.DESCRIPTION, f"STQFT, simulated, mrot={mrot}, post processed")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

    pt += 1
    mrot = PI/2**(nQubits-pt)

plotData = grader_inst.show(subplot=[2,nQubits+2,nQubits])

exp = export(topic=TOPIC, identifier="grader_stqft_sim")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, stqft_sim")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Real STQFT")

mrot = 0
pt = 0
grader_inst = grader()
device = "ibmq_melbourne"
_, backend = loadBackend(simulation=True, backendName=device)

while mrot <= PI/2:
    stqft = transform(stqft_framework, minRotation=mrot, suppressPrint=True, reuseBackend=backend)
    y_hat_real, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
    y_hat_real_p, f_p, t_p = stqft.postProcess(y_hat_real, f ,t, scale='mel')
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = stqft.show(y_hat_real_p, f_p, t_p, subplot=[2,nQubits+2,pt+nQubits+5], title=f"STQFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)

    snr = grader_inst.calculateNoisePower(y_hat_real_p, y_hat_stft_p)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")

    exp = export(topic=TOPIC, identifier=f"stqft_real_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_real_p)
    exp.setData(export.DESCRIPTION, f"STQFT, simulated, {device} noise, mrot={mrot}, post processed")
    exp.setData(export.BACKEND, backend)
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

    pt += 1
    mrot = PI/2**(nQubits-pt)

plotData = grader_inst.show(subplot=[2,nQubits+2,2*(nQubits+3)])

exp = export(topic=TOPIC, identifier="grader_stqft_real")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, stqft_real")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()


print("Showing all figures")
frontend.primeTime() # Show all with blocking