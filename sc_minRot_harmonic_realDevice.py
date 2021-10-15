from math import exp
from qft import qft_framework, loadBackend
from fft import fft_framework
from qft import setupMeasurementFitter, loadBackend, loadNoiseModel
from frontend import frontend, grader, signal, transform, export
from utils import PI

frontend.enableInteractive()
TOPIC = "minRot_harmonic_realDevice_multiExp"
export.checkWorkingTree()
device = "ibmq_quito"

nQubits = 4
rows = 3
nExp=5

print("Initializing Harmonic Signal")

y = signal(samplingRate=1000, amplification=1, duration=0, nSamples=2**nQubits)

y.addFrequency(250) # you should choose a frequency matching a multiple of samplingRate/nSamples
y.addFrequency(125) # you should choose a frequency matching a multiple of samplingRate/nSamples

plotData = y.show(subplot=[rows,nQubits+2,1], title='signal')

exp = export(topic=TOPIC, identifier="signal")
# exp.setData(export.SIGNAL, y)
exp.setData(export.DESCRIPTION, "Harmonic Signal, 125 and 250 Hz at 1kHz, 2^4 samples")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing FFT")

fft = transform(fft_framework)
y_hat, f = fft.forward(y)
y_hat_ideal_p, f_p = fft.postProcess(y_hat, f)
plotData = fft.show(y_hat_ideal_p, f_p, subplot=[rows,nQubits+2,nQubits+3], title="FFT (ref)")

exp = export(topic=TOPIC, identifier="fft")
exp.setData(export.SIGNAL, y_hat_ideal_p)
exp.setData(export.DESCRIPTION, "FFT, default param, post processed")
exp.setData(export.PLOTDATA, plotData)
exp.doExport()

print("Processing Simulated QFT")



# for expIt in range(0, nExp):
#     print(f"Experiment {expIt}\n")
#     mrot = 0
#     pt = 0
#     grader_inst = grader()

#     while mrot <= PI/2:
#         qft = transform(qft_framework, minRotation=mrot, suppressPrint=True)
#         y_hat, f = qft.forward(y)
#         y_hat_sim_p, f_p = qft.postProcess(y_hat, f)
#         ylabel = "Amplitude" if pt == 0 else " "
#         plotData = qft.show(y_hat_sim_p, f_p, subplot=[rows,nQubits+2,pt+2], title=f"QFT_sim, mr:{mrot:.2f}", xlabel=" ", ylabel=ylabel)

#         snr = grader_inst.calculateNoisePower(y_hat_sim_p, y_hat_ideal_p)
#         print(f"Calculated an snr of {snr} db")
#         grader_inst.log(snr, mrot)
#         print(f"Minimum rotation is: {mrot}")

#         exp = export(topic=TOPIC, identifier=f"{expIt}_qft_sim_mr_{mrot:.2f}")
#         exp.setData(export.SIGNAL, y_hat_sim_p)
#         exp.setData(export.DESCRIPTION, f"QFT, simulated, mrot={mrot}, post processed. Experiment {expIt}")
#         exp.setData(export.BACKEND, qft.transformation.getBackend())
#         exp.setData(export.PLOTDATA, plotData)
#         exp.doExport()

#         pt += 1
#         mrot = PI/2**(nQubits-pt)
#     mrot = 0
#     pt = 0

#     plotData = grader_inst.show(subplot=[rows,nQubits+2,nQubits+2])

#     exp = export(topic=TOPIC, identifier=f"{expIt}_grader_qft_sim")
#     exp.setData(export.GRADERX, grader_inst.xValues)
#     exp.setData(export.GRADERY, grader_inst.yValues)
#     exp.setData(export.DESCRIPTION, f"Grader, qft_sim. Experiment {expIt}")
#     exp.setData(export.PLOTDATA, plotData)
#     exp.doExport()

print("Processing Simulated QFT with Noise")

_, backendInstance = loadBackend(simulation=True, backendName=device)
_, noiseModel = loadNoiseModel(backendName=backendInstance)

for expIt in range(0, nExp):
    print(f"Experiment {expIt}\n")
    mrot = 0
    pt = 0
    grader_inst = grader()

    while mrot <= PI/2:
        qft = transform(qft_framework, minRotation=mrot, suppressPrint=True, simulation=True, useNoiseModel=True, noiseModel=noiseModel, backend=backendInstance)

        y_hat, f = qft.forward(y)
        y_hat_sim_n_p, f_p = qft.postProcess(y_hat, f)
        ylabel = "Amplitude" if pt == 0 else " "
        # 2nd entry in 2nd row: 0+1+(nQubits+3)+1)
        plotData = qft.show(y_hat_sim_n_p, f_p, subplot=[rows,nQubits+2,pt+nQubits+4], title=f"QFT_sim_n, mr:{mrot:.2f}",  xlabel=" ", ylabel=ylabel)
        
        snr = grader_inst.calculateNoisePower(y_hat_sim_n_p, y_hat_ideal_p)
        print(f"Calculated an snr of {snr} db")
        grader_inst.log(snr, mrot)
        print(f"Minimum rotation is: {mrot}")

        exp = export(topic=TOPIC, identifier=f"{expIt}_qft_sim_n_mr_{mrot:.2f}")
        exp.setData(export.SIGNAL, y_hat_sim_n_p)
        exp.setData(export.DESCRIPTION, f"QFT, real Device, {device} noise, mrot={mrot}, post processed. Experiment {expIt}")
        exp.setData(export.BACKEND, qft.transformation.getBackend())
        exp.setData(export.PLOTDATA, plotData)
        exp.doExport()

        pt += 1
        mrot = PI/2**(nQubits-pt)

    plotData = grader_inst.show(subplot=[rows,nQubits+2,2*(nQubits+2)])

    exp = export(topic=TOPIC, identifier=f"{expIt}_grader_qft_sim_n")
    exp.setData(export.GRADERX, grader_inst.xValues)
    exp.setData(export.GRADERY, grader_inst.yValues)
    exp.setData(export.DESCRIPTION, f"Grader, qft_sim_n. Experiment {expIt}")
    exp.setData(export.PLOTDATA, plotData)
    exp.doExport()

# print("Processing Real QFT")


# # _, backend = loadBackend(simulation=False, backendName=device)

# for expIt in range(0, nExp):
#     print(f"Experiment {expIt}\n")
#     mrot = 0
#     pt = 0
#     grader_inst = grader()

#     while mrot <= PI/2:
#         qft = transform(qft_framework, minRotation=mrot, suppressPrint=False, simulation=False, backend=device)

#         y_hat, f = qft.forward(y)
#         y_hat_real_p, f_p = qft.postProcess(y_hat, f)
#         ylabel = "Amplitude" if pt == 0 else " "
#         # 2nd entry in 2nd row: 0+1+(nQubits+3)+1)
#         plotData = qft.show(y_hat_real_p, f_p, subplot=[rows,nQubits+2,pt+2*nQubits+6], title=f"QFT_real, mr:{mrot:.2f}",  xlabel="Freq (Hz)", ylabel=ylabel)
        
#         snr = grader_inst.calculateNoisePower(y_hat_real_p, y_hat_ideal_p)
#         print(f"Calculated an snr of {snr} db")
#         grader_inst.log(snr, mrot)
#         print(f"Minimum rotation is: {mrot}")

#         exp = export(topic=TOPIC, identifier=f"{expIt}_qft_real_mr_{mrot:.2f}")
#         exp.setData(export.SIGNAL, y_hat_real_p)
#         exp.setData(export.DESCRIPTION, f"QFT, real Device, {device} noise, mrot={mrot}, post processed. Experiment {expIt}")
#         exp.setData(export.BACKEND, qft.transformation.getBackend())
#         exp.setData(export.PLOTDATA, plotData)
#         exp.doExport()

#         pt += 1
#         mrot = PI/2**(nQubits-pt)

#     plotData = grader_inst.show(subplot=[rows,nQubits+2,3*(nQubits+2)])

#     exp = export(topic=TOPIC, identifier=f"{expIt}_grader_qft_real")
#     exp.setData(export.GRADERX, grader_inst.xValues)
#     exp.setData(export.GRADERY, grader_inst.yValues)
#     exp.setData(export.DESCRIPTION, f"Grader, qft_real. Experiment {expIt}")
#     exp.setData(export.PLOTDATA, plotData)
#     exp.doExport()

print("Showing all figures")
frontend.primeTime() # Show all with blocking