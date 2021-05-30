import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
plt.ion()

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

    def __init__(self, samplingRate=40, amplification=1, duration=1, nSamples=None) -> None:
        self.amplification = amplification
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate

        t_max = min(duration, nSamples/samplingRate)
        self.t = np.arange(0,t_max,self.samplingInterval)
        
        self.y = np.zeros(self.t.size)

    def addFrequency(self, frequency, phase=0):
        self.frequencies.append(frequency)
        self.phases.append(phase)
        
    def sample(self):
        self.y = np.zeros(self.t.size)
        for frequency, phase in zip(self.frequencies, self.phases):
            self.y += self.amplification*np.sin(2*np.pi*frequency*self.t-phase)

        return self.y
        
    # def sample(self, frequency=2.):
    #     self.frequency = frequency
    #     self.y = np.sin(2*np.pi*self.frequency*self.t)

    #     return self.y
    
    def show(self, path=None):
        plt.figure(figsize = (10, 6))
        plt.plot(self.t, self.y, 'r')
        plt.ylabel('Amplitude')
        plt.xlabel('Normalized Time')

        plt.show()

        if path != None:
            plt.savefig(path)

class transform():
    def __init__(self, transformation) -> None:
        self.transformation = transformation()

    def forward(self, y):
        y_hat = self.transformation.transform(y)

        n = np.arange(len(y_hat))
        T = len(y_hat)/y.samplingRate
        f = n/T

        return y_hat, f

    def show(self, y_hat, f, path=None):
        n_oneside = len(y_hat)//2
        # get the one side frequency
        f_oneside = f[:n_oneside]

        # normalize the amplitude
        y_hat_oneside =y_hat[:n_oneside]/n_oneside

        plt.figure(figsize = (10, 6))
        # plt.subplot(121)
        plt.stem(f_oneside, abs(y_hat_oneside), 'b', markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('Amplitude |y_hat(freq)|')

        # plt.subplot(122)
        # plt.stem(f_oneside, abs(y_hat_oneside), 'b', markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.xlim(0, 10)
        # plt.tight_layout()
        # plt.show()

        if path != None:
            plt.savefig(path)

def primeTime():
    plt.ioff()
    input("Press any key to close all figures")
    plt.close('all')