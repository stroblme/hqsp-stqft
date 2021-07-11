from math import exp
from qft import qft_framework, loadBackend
from fft import fft_framework
from frontend import frontend, grader, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "minRot_harmonic"
export.checkWorkingTree()

nQubits = 4

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nQubits)

y.addFrequency(250) # you should choose a frequency matching a multiple of samplingRate/nSamples
y.addFrequency(125) # you should choose a frequency matching a multiple of samplingRate/nSamples

plotData = y.show(subplot=[2,nQubits+3,1], title='signal')

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Harmonic Signal, 125 and 250 Hz at 1kHz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
y_hat_ideal_p, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal_p, f_p, subplot=[2,nQubits+3,nQubits+4], title="FFT (ref)")

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal_p)
exp.setData(export.DESCRIPTION, "FFT, default param, post processed")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Simulated QFT")

mrot = 0
pt = 0
grader_inst = grader()

while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False)
    y_hat, f = qft.forward(y)
    y_hat_sim_p, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    plotData = qft.show(y_hat_sim_p, f_p, subplot=[2,nQubits+3,pt+2], title=f"QFT_sim, mr:{mrot:.2f}", xlabel=" ", ylabel=ylabel)

    snr = grader_inst.calculateNoisePower(y_hat_sim_p, y_hat_ideal_p)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")

    exp = export(topic=TOPIC, identifier=f"qft_sim_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_sim_p)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, mrot={mrot}, post processed")
    exp.setData(export.BACKEND, qft.transformation.getBackend())
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

    pt += 1
    mrot = PI/2**(nQubits+1-pt)

plotData = grader_inst.show(subplot=[2,nQubits+3,nQubits+3])

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

device = "ibmq_quito"
_, backend = loadBackend(simulation=True, backendName=device)

while mrot <= PI/2:
    qft = transform(qft_framework, minRotation=mrot, suppressPrint=False, simulation=True, backendName=device, reuseBackend=backend)

    y_hat, f = qft.forward(y)
    y_hat_real_p, f_p = qft.postProcess(y_hat, f)
    ylabel = "Amplitude" if pt == 0 else " "
    # 2nd entry in 2nd row: 0+1+(nQubits+3)+1)
    plotData = qft.show(y_hat_real_p, f_p, subplot=[2,nQubits+3,pt+nQubits+5], title=f"QFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)
    
    snr = grader_inst.calculateNoisePower(y_hat_real_p, y_hat_ideal_p)
    print(f"Calculated an snr of {snr} db")
    grader_inst.log(snr, mrot)
    print(f"Minimum rotation is: {mrot}")

    exp = export(topic=TOPIC, identifier=f"qft_real_mr_{mrot:.2f}")
    exp.setData(export.SIGNAL, y_hat_real_p)
    exp.setData(export.DESCRIPTION, f"QFT, simulated, {device} noise, mrot={mrot}, post processed")
    exp.setData(export.BACKEND, qft.transformation.getBackend())
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

    pt += 1
    mrot = PI/2**(nQubits+1-pt)

plotData = grader_inst.show(subplot=[2,nQubits+3,2*(nQubits+3)])

exp = export(topic=TOPIC, identifier="grader_qft_real")
exp.setData(export.GRADERX, grader_inst.xValues)
exp.setData(export.GRADERY, grader_inst.yValues)
exp.setData(export.DESCRIPTION, "Grader, qft_real")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()



print("Showing all figures")
frontend.primeTime() # Show all with blocking