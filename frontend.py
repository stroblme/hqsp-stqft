import numpy as np
from numpy import pi
from scipy import signal as scipySignal
import matplotlib.pyplot as plt
from math import log, ceil, floor

plt.style.use('seaborn-poster')

def enableInteractive():
    global plt
    plt.ion()

def disableInteractive():
    global plt
    plt.ioff()
    # frequencies = list()

    # def __init__(self, samplingRate=100) -> None:
    #     self.samplingRate = samplingRate
    #     self.samplingInterval = 1/self.samplingRate
    #     self.t = np.arange(0,1,self.samplingInterval)
        
    #     self.y = np.zeros(self.t.size)

    # def addFrequency(self, frequency):
    #     self.frequencies.append(frequency)
        
    # def sample(self):
    #     for frequency in self.frequencies:
    #         self.y += np.sin(2*np.pi*frequency*self.t)

    #     return self.y

class signal():
    frequencies = list()
    phases = list()

    def __init__(self, samplingRate=40, amplification=1, duration=2, nSamples=80, signalType='sin') -> None:
        """Signal Init

        Args:
            samplingRate (int, optional): [description]. Defaults to 40.
            amplification (int, optional): [description]. Defaults to 1.
            duration (int, optional): Duration of the created signal. Defaults to 2.
            nSamples ([type], optional): Sample length of the signal. Defaults to 80.
        """
        # Set the class attributes
        self.amplification = amplification
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate

        self.signalType = signalType
        
        # Either use the duration or the number of samples depending on what's longer
        t_max = max(duration, nSamples*self.samplingInterval)

        # Get the closest min. int which is a power of 2
        nSamples = int(t_max/self.samplingInterval)
        nSamples_log2_min = floor(log(nSamples, 2))

        # Update the number of samples and the duration based on the previous modifications
        self.nSamples = 2**nSamples_log2_min
        self.duration = self.nSamples*self.samplingRate
        t_max = self.nSamples*self.samplingInterval

        print(f"Signal duration set to {t_max}")

        # Create time vector
        self.t = np.arange(0,t_max,self.samplingInterval)
        
        # Create the signal
        self.y = np.zeros(self.nSamples)

    def addFrequency(self, frequency, phase=0):
        if frequency > self.samplingRate/2:
            print("WARNING: Nyquist not fulfilled!")
            
        self.frequencies.append(frequency)
        self.phases.append(phase)
        
    def sample(self):
        self.y = np.zeros(self.nSamples)
        if self.signalType == 'sin':
            for frequency, phase in zip(self.frequencies, self.phases):
                self.y += self.amplification*np.sin(2*np.pi*frequency*self.t-phase)
        elif self.signalType == 'chirp':
            f0 = -1
            f1 = -1

            for frequency, phase in zip(self.frequencies, self.phases):
                if f0 == -1:
                    f0 = frequency
                elif f1 == -1 and f0 != -1:
                    f1 = frequency
                if f0 != -1 and f1 != -1:
                    self.y += self.amplification*scipySignal.chirp(self.t, f0=f0, f1=f1, t1=phase, method='linear')
                    f0 = -1
                    f1 = -1
        else:
            print('Must be either sin, chirp')
        return self.y
    
    def show(self, path=None):
        self.sample()

        minF = min(self.frequencies)
        maxP = max(self.phases)
        maxT = (1/minF + maxP)*2
        minSamples = int(maxT*self.samplingRate)

        plt.figure(figsize = (10, 6))
        plt.plot(self.t[:minSamples], self.y[:minSamples], 'r')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (excerp) [s]')
        plt.title(type(self).__name__)


        plt.show()

        if path != None:
            plt.savefig(path)

class transform():
    def __init__(self, transformation, **kwargs):
        self.transformation = transformation(**kwargs)

    def forward(self, y):
        y_hat = self.transformation.transform(y)

        n = np.arange(len(y_hat))
        T = len(y_hat)/y.samplingRate
        f = n/T

        return y_hat, f

    def show(self, y_hat, f, isOneSided=False, path=None):
        if not isOneSided:
            n = len(y_hat)//2
            # get the one side frequency
            f = f[:n]

            # normalize the amplitude
            y_hat =y_hat[:n]/n

        plt.figure(figsize = (10, 6))
        # plt.subplot(121)
        plt.stem(f, abs(y_hat), 'b', markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amplitude (abs)')
        plt.title(type(self.transformation).__name__)

        # plt.subplot(122)
        # plt.stem(f, abs(y_hat), 'b', markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.xlim(0, 10)
        # plt.tight_layout()
        plt.show()

        if path != None:
            plt.savefig(path)

def primeTime():
    disableInteractive()
    input("Press any key to close all figures\n")
    plt.close('all')