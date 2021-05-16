import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')
plt.ion()

class signal():
    def __init__(self, samplingRate=100) -> None:
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate
        self.t = np.arange(0,1,self.samplingInterval)
        
        self.sample()
        
    def sample(self, frequency=2.):
        self.frequency = frequency
        self.y = np.sin(2*np.pi*self.frequency*self.t)

        return self.y
    
    def show(self):
        plt.figure(figsize = (8, 6))
        plt.plot(self.t, self.y, 'r')
        plt.ylabel('Amplitude')
        plt.xlabel('Normalized Time')

        plt.show()

class transform():
    def __init__(self, transformation) -> None:
        self.transformation = transformation

    def forward(self, y):
        y_hat = self.transformation(y.sample())

        n = np.arange(len(y_hat))
        T = len(y_hat)/y.samplingRate
        f = n/T

        return y_hat, f

    def show(self, y_hat, f):
        n_oneside = len(y_hat)//2
        # get the one side frequency
        f_oneside = f[:n_oneside]

        # normalize the amplitude
        y_hat_oneside =y_hat[:n_oneside]/n_oneside

        plt.figure(figsize = (12, 6))
        plt.subplot(121)
        plt.stem(f_oneside, abs(y_hat_oneside), 'b',          markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('DFT Amplitude |X(freq)|')

        plt.subplot(122)
        plt.stem(f_oneside, abs(y_hat_oneside), 'b',          markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.xlim(0, 10)
        plt.tight_layout()
        plt.show()

def primeTime():
    plt.ioff()
    plt.show()