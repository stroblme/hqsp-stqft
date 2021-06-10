import numpy as np
from numpy import pi
from scipy import signal as scipySignal
import matplotlib.pyplot as plt
from math import log, ceil, floor
from copy import deepcopy

import librosa

def enableInteractive():
    global plt
    plt.ion()

def disableInteractive():
    global plt
    plt.ioff()

def setStylesheet(theme):
    plt.style.use(theme)

class signal():
    frequencies = list()
    phases = list()

    def __init__(self, samplingRate=40, amplification=1, duration=2, nSamples=80, signalType='sin', path='') -> None:
        """Signal Init

        Args:
            samplingRate (int, optional): [description]. Defaults to 40.
            amplification (int, optional): [description]. Defaults to 1.
            duration (int, optional): Duration of the created signal. Defaults to 2.
            nSamples ([type], optional): Sample length of the signal. Defaults to 80.
        """
        # Set the class attributes
        self.amplification = amplification
        self.setSamplingRate(samplingRate)

        self.signalType = signalType
        
        # Set the number of samples based on duration and target num of samples such that it matches 2**n
        self.setNSamples(duration, nSamples)


        if signalType=='file':
            assert path!=''
            self.loadFile(path)

            self.lockSampling = True
        else:
            # Create the signal
            self.createEmptySignal(self.nSamples)

            self.lockSampling = False

        print(f"Signal duration set to {self.duration}s, resulting in {self.nSamples} samples")
        print(f"Sampling Rate is {self.samplingRate} with an amplification of {self.amplification}")
        self.t = np.arange(0,self.duration,self.samplingInterval)

    def createEmptySignal(self, nSamples):
        self.y = np.zeros(nSamples)


    def loadFile(self, path, zeroPadding=True):

        samplingRate = librosa.get_samplerate(path)
        if samplingRate < self.samplingRate:
            print(f'Warning: provided sampling rate ({self.samplingRate}) is higher than the one of the audio ({samplingRate}). Will upsample.')
        elif samplingRate > self.samplingRate:
            print(f'Warning: provided sampling rate ({self.samplingRate}) is lower than the one of the audio ({samplingRate}). Will downsample.')
        
        duration = librosa.get_duration(filename=path)
        if duration < self.duration:
            if zeroPadding:
                print(f'Audio is not long enough ({duration}). Will use zero-padding to fill up distance to {self.duration}')

                self.createEmptySignal(self.nSamples)

                y_p, _ = librosa.load(path, sr=self.samplingRate)

                self.y[0:y_p.size] = y_p
                return
            else:
                self.setNSamples(duration=duration, nSamples=0)

        self.y = librosa.load(path, sr=self.samplingRate, duration=self.duration)


        # mel_feat = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
        # all_wave.append(np.expand_dims(mel_feat, axis=2))
        # all_label.append(label)

    def setSamplingRate(self, samplingRate):
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate

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

    def split(self, nSamplesWindow, overlapFactor=0, windowType=None):
        self.sample()

        if windowType == 'hanning':
            window = np.hanning(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hanning window")
        elif windowType == 'hamming':
            window = np.hamming(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hamming window")
        else:
            window = 1.

        hopSize = np.int32(np.floor(nSamplesWindow * (1-overlapFactor)))
        nParts = np.int32(np.ceil(len(self.y) / np.float32(hopSize)))
        
        y_split_list = list()

        for i in range(0,nParts-1): # -1 because e.g with an overlap of 0.5 we will get 2*N - 1 segments
            currentHop = hopSize * i                        # figure out the current segment offset
            segment = self.y[currentHop:currentHop+nSamplesWindow]  # get the current segment
            windowed = segment * window                       # multiply by the half cosine function
            
            y = deepcopy(self)
            y.externalSample(windowed, self.t[currentHop:currentHop+nSamplesWindow])
            y_split_list.append(y)

        print(f"Signal divided into {nParts-1} parts with a window length of {nSamplesWindow} each")


        return y_split_list

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
    
    def show(self, subplot=None, path=None, ignorePhaseShift=False):

        if self.signalType=='file':
            minSamples = self.y.size-1 # Use all samples
        else:
            self.sample()   # Only sample if not file, as data is assumed to be loaded
            minF = min(self.frequencies)
            maxP = max(self.phases) if not ignorePhaseShift else 0
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

    def forward(self, y, **kwargs):
        y_hat = self.transformation.transform(y, **kwargs)

        n = np.arange(y_hat.shape[0])
        F = y_hat.shape[0]/y.samplingRate
        f = n/F
        
        if len(y_hat.shape) == 2:
            n = np.arange(y_hat.shape[1])
            T = y_hat.shape[1]/y.duration
            t = n/T
            return y_hat, f, t
        else:
            return y_hat, f

    def swapaxes(self, y_hat):
        return np.swapaxes(y_hat, 0, 1)

    def show(self, y_hat, f, t=None, scale=None, autopower=True, normalize=True, fmax=None, subplot=None, path=None):

        n = y_hat.shape[0]//2
        # get the one side frequency
        f = f[:n] if autopower else f

        # normalize the amplitude
        y_hat = np.abs(y_hat * np.conj(y_hat))
        if autopower:
            y_hat =(y_hat[:n]/n if t is None else y_hat[:n,:]/n) 

        if fmax != None:
            if fmax > f.max():
                print(f"f_max {fmax} is not lower than the actual max frequency {f.max()}")
            else:
                f_idx = int(np.where(f>fmax)[0][0])
                f = f[:f_idx]
                y_hat = y_hat[:f_idx,:]    


        if normalize:
            y_hat = y_hat*(1/y_hat.max())

        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
            plt.subplots_adjust(wspace=0.58)
        else:
            plt.figure(figsize = (10, 6))

        if scale == 'log':
            y_hat = 20*np.log10(y_hat)
            plt.yscale('log')
        elif scale == 'mel':
            y_hat = 1127*np.log10(1+y_hat/700) # mel scale formula
            plt.yscale('log')

        if t is None:
            plt.stem(f, np.abs(y_hat), 'b', markerfmt=" ", basefmt="-b")
            plt.xlabel('Freq [Hz]')
            plt.ylabel('Amplitude (abs)')
        else:
            plt.pcolormesh(t, f, np.abs(y_hat), cmap='cividis', shading='auto')
            plt.xlabel('Time [s]')
            plt.ylabel('Freq [Hz]')
            # plt.colorbar(format='%+2.0f')
                
        plt.title(type(self.transformation).__name__)

        if path is not None:
            plt.savefig(path)

def primeTime():
    plt.show()
    disableInteractive()
    input("Press any key to close all figures\n")
    plt.close('all')