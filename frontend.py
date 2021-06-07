import numpy as np
from numpy import pi
from scipy import signal as scipySignal
import matplotlib.pyplot as plt
from math import log, ceil, floor
from copy import deepcopy

plt.style.use('seaborn-poster')

def enableInteractive():
    global plt
    plt.ion()

def disableInteractive():
    global plt
    plt.ioff()

def setStylesheet(theme):
    plt.style.use('seaborn-poster')

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
        
        self.setNSamples(duration, nSamples)
        print(f"Signal duration set to {self.duration}, resulting in {self.nSamples} samples")
        print(f"Sampling Rate is {self.samplingRate} with an amplification of {self.amplification}")

        # Create time vector
        self.t = np.arange(0,self.duration,self.samplingInterval)
        
        # Create the signal
        self.y = np.zeros(self.nSamples)

        self.lockSampling=False

    def setNSamples(self, duration=2, nSamples=80):
        # Either use the duration or the number of samples depending on what's longer
        t_max = max(duration, nSamples*self.samplingInterval)

        # Get the closest min. int which is a power of 2
        nSamples = int(t_max/self.samplingInterval)
        nSamples_log2_min = floor(log(nSamples, 2))

        # Update the number of samples and the duration based on the previous modifications
        self.nSamples = 2**nSamples_log2_min
        self.duration = self.nSamples*self.samplingInterval

        return self.duration

    def addFrequency(self, frequency, phase=0):
        if frequency > self.samplingRate/2:
            print("WARNING: Nyquist not fulfilled!")
            
        self.frequencies.append(frequency)
        self.phases.append(phase)

    def externalSample(self, y, t):
        self.y = y
        self.t = t
        self.setNSamples(0,t.size)
        self.lockSampling=True

    def split(self, nParts):
        self.sample()

        if self.nSamples%nParts != 0:
            raise RuntimeError(f"Signal with length {self.nSamples} cannot be splitted in {nParts} parts")

        y_split = np.array_split(self.y, nParts)
        t_split = np.array_split(self.t, nParts)
        
        y_split_array = list()

        for i in range(0,nParts):
            y = deepcopy(self)
            y.externalSample(y_split[i], t_split[i])
            y_split_array.append(y)

        return y_split_array

    def sample(self):
        if self.lockSampling:
            return self.y

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
    
    def show(self, subplot=None, path=None):
        self.sample()

        minF = min(self.frequencies)
        maxP = max(self.phases)
        maxT = (1/minF + maxP)*2
        minSamples = int(maxT*self.samplingRate)

        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
        else:
            plt.figure(figsize = (10, 6))

        plt.plot(self.t[:minSamples], self.y[:minSamples], 'r')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (excerp) [s]')
        plt.title(type(self).__name__)


        if path is not None:
            plt.savefig(path)

class transform():
    def __init__(self, transformation, **kwargs):
        self.transformation = transformation(**kwargs)

    def forward(self, y, *kwargs):
        y_hat = self.transformation.transform(y, *kwargs)

        n = np.arange(len(y_hat))
        T = len(y_hat)/y.samplingRate
        f = n/T

        return y_hat, f

    def show(self, y_hat, f, isOneSided=False, subplot=None, path=None):
        if not isOneSided:
            n = len(y_hat)//2
            # get the one side frequency
            f = f[:n]

            # normalize the amplitude
            y_hat =y_hat[:n]/n

        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
        else:
            plt.figure(figsize = (10, 6))

        plt.stem(f, abs(y_hat), 'b', markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amplitude (abs)')
        plt.title(type(self.transformation).__name__)

        if path is not None:
            plt.savefig(path)

def primeTime():
    plt.show()
    disableInteractive()
    input("Press any key to close all figures\n")
    plt.close('all')