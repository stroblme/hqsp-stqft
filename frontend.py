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
        """Signal Init

        Args:
            samplingRate (int, optional): [description]. Defaults to 40.
            amplification (int, optional): [description]. Defaults to 1.
            duration (int, optional): Duration of the created signal. Defaults to 1.
            nSamples ([type], optional): Sample length of the signal. Defaults to None.
        """
        
        self.amplification = amplification
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate
        self.duration = duration
        
        t_max = min(duration, nSamples/samplingRate)
        
        self.t = np.arange(0,t_max,self.samplingInterval)
        self.nSamples = self.t.size
        
        self.y = np.zeros(self.nSamples)

    def addFrequency(self, frequency, phase=0):
        self.frequencies.append(frequency)
        self.phases.append(phase)
        
    def sample(self):
        self.y = np.zeros(self.nSamples)
        for frequency, phase in zip(self.frequencies, self.phases):
            self.y += self.amplification*np.sin(2*np.pi*frequency*self.t-phase)

        return self.y
    
    def show(self, path=None):
        minF = min(self.frequencies)
        maxT = 1/minF
        minSamples = int(maxT*self.samplingRate)

        plt.figure(figsize = (10, 6))
        plt.plot(self.t[:minSamples], self.y[:minSamples], 'r')
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.title(type(self).__name__)


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
    plt.ioff()
    input("Press any key to close all figures\n")
    plt.close('all')