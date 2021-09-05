from stft import stft_framework
from stqft import stqft_framework
from frontend import frontend, signal, transform

from tests import *

frontend.enableInteractive()
TOPIC = "speech"

# speechSignal = '../dataset/zero/4a1e736b_nohash_2.wav' #male clear
# speechSignal = '../dataset/zero/0fa1e7a9_nohash_1.wav' #male noise
# speechSignal = '../dataset/zero/7ea032f3_nohash_3.wav'  #male questionary
# speechSignal = '../dataset/zero/8e05039f_nohash_4.wav'  #female clear
speechSignal = '../dataset/zero/4634529e_nohash_2.wav'  #female noise


windowLength = 2**10
overlapFactor=0.5
windowType='hanning'

print("Initializing Signal")

y = signal(samplingRate=16000, signalType='file', path=speechSignal)
y.show(subplot=[1,3,1])

print("Processing STFT")
stft = transform(stft_framework)
y_hat_stft, f ,t = stft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stft_p, f_p, t_p = stft.postProcess(y_hat_stft, f ,t, scale='mel', fmax=4000)
stft.show(y_hat_stft_p, f_p, t_p, subplot=[1,3,2])


print("Processing STQFT")
stqft = transform(stqft_framework, suppressPrint=True, minRotation=0.2)
y_hat_stqft, f, t = stqft.forward(y, nSamplesWindow=windowLength, overlapFactor=overlapFactor, windowType=windowType)
y_hat_stqft_p, f_p, t_p = stqft.postProcess(y_hat_stqft, f ,t, scale='mel', fmax=4000)
stqft.show(y_hat_stqft_p, f_p, t_p, subplot=[1,3,3])

# grader_inst = grader()
# y_hat_diff = grader_inst.correlate2d(y_hat_stft_p, y_hat_stqft_p)
# grader_inst.show(y_hat_diff, f_p, t=t_p, subplot=[1,4,4])


print("Showing all figures")
frontend.primeTime() # Show all with blocking