import numpy as np
from numpy import pi
from scipy import signal as scipySignal
import matplotlib.pyplot as plt
from math import log, ceil, floor, sqrt
from copy import deepcopy
import os
import pickle
import git

import librosa

from qbstyles import mpl_style

COLORMAP = 'plasma'
SHADING='nearest'
DARK=True


def enableInteractive():
    global plt
    plt.ion()

def disableInteractive():
    global plt
    plt.ioff()

def setTheme(dark=False):
    DARK = dark

    mpl_style(dark=DARK, minor_ticks=False)

mpl_style(dark=DARK, minor_ticks=False)
# plt.style.use('./styles/.mplstyle')

class signal():
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
        self.frequencies = list()
        self.phases = list()

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
        """Loads an audio sample from file

        Args:
            path ([type]): [description]
            zeroPadding (bool, optional): [description]. Defaults to True.
        """
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

                self.y[0:y_p.size] = y_p * self.amplification
                return
            else:
                self.setNSamples(duration=duration, nSamples=0)

        self.y = librosa.load(path, sr=self.samplingRate, duration=self.duration) * self.amplification


        # mel_feat = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
        # all_wave.append(np.expand_dims(mel_feat, axis=2))
        # all_label.append(label)

    def setSamplingRate(self, samplingRate):
        """Sets the sampling rate for the current signal instance

        Args:
            samplingRate ([type]): [description]
        """
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

        if windowType == 'hann':
            window = np.hanning(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hanning window")
        elif windowType == 'hamm':
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
    
    def show(self, subplot=None, ignorePhaseShift=False, xlabel="Time (s)", ylabel="Amplitude"):

        fighandle = plt.figure()

        if self.signalType=='file':
            minSamples = self.y.size-1 # Use all samples
        else:
            self.sample()   # Only sample if not file, as data is assumed to be loaded
            minF = min(self.frequencies)
            maxP = max(self.phases) if not ignorePhaseShift else 0
            maxT = (1/minF + maxP)*2
            minSamples = int(maxT*self.samplingRate)

        if subplot is not None:
            plt.subplot(*subplot, frameon=False)
        else:
            plt.figure(figsize = (10, 6))

        plt.plot(self.t[:minSamples], self.y[:minSamples], '.-')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(type(self).__name__)
        plt.tight_layout()


        return fighandle

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

    def backward(self, y_hat, **kwargs):
        y = self.transformation.transformInv(y_hat, **kwargs)



    def postProcess(self, y_hat, f, t=None, scale=None, autopower=True, normalize=True, fmax=None):
        # get the one side frequency
        if autopower:
            y_hat = np.abs(y_hat)
            n = y_hat.shape[0]//2
            f = f[:n]
            y_hat =(y_hat[:n]/n if t is None else y_hat[:n,:]/n) 

        if fmax != None:
            if fmax >= f.max():
                print(f"f_max {fmax} is not lower than the actual max frequency {f.max()}")
            else:
                f_idx = int(np.where(f>fmax)[0][0])
                f = f[:f_idx]
                y_hat = y_hat[:f_idx,:]    

        if normalize:
            y_hat = y_hat*(1/y_hat.max())
            # y_hat = y_hat*(1/sqrt(y_hat.shape[0]))

        if scale == 'log':
            y_hat = 20*np.log10(y_hat)
            # plt.yscale('log',base=2)
        elif scale == 'mel':
            y_hat = 1127*np.log10(1+y_hat/700) # mel scale formula
            # plt.yscale('log',base=2)

        if t is None:
            return y_hat, f
        else:
            return y_hat, f, t

    def swapaxes(self, y_hat):
        return np.swapaxes(y_hat, 0, 1)

    def show(self, y_hat, f, t=None, subplot=None, title="", xlabel='', ylabel=''):
        fighandle = plt.figure()

        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
            plt.subplots_adjust(wspace=0.58)
        else:
            plt.figure(figsize = (10, 6))

        if t is None:
            plt.stem(f, np.abs(y_hat))
            if xlabel != "":
                plt.xlabel(xlabel)
            else:
                plt.xlabel('Freq (Hz)')
            if ylabel != "":
                plt.ylabel(ylabel)
            else:
                plt.ylabel('Amplitude (abs)')
        else:
            plt.pcolormesh(t, f, y_hat, cmap=COLORMAP, shading=SHADING)
            if xlabel != "":
                plt.xlabel(xlabel)
            else:
                plt.xlabel('Time (s)')
            if ylabel != "":
                plt.ylabel(ylabel)
            else:
                plt.ylabel('Freq (Hz)')
            # plt.colorbar(format='%+2.0f')
                
        if title!="":
            plt.title(title)
        else:
            plt.title(type(self.transformation).__name__)

        plt.tight_layout()
        
        return fighandle

class grader():
    epsilon=1e-10
    def __init__(self):
        self.yValues = np.array([])
        self.xValues = np.array([])

    def correlate2d(self, a, b):
        y_hat_diff = scipySignal.correlate2d(a, b, mode='same')

        return y_hat_diff

    def calculateNoisePower(self, y, y_ref):
        diff = np.abs(y-y_ref)

        snr = np.divide(np.sum(np.abs(y_ref)),np.sum(diff)+self.epsilon)

        return 10*np.log10(snr)

    def log(self, ylabel, xlabel):
        self.yValues = np.append(self.yValues, [ylabel])
        self.xValues = np.append(self.xValues, [xlabel])    

    def show(self, subplot=None):
        
        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
            plt.subplots_adjust(wspace=0.58)
        else:
            plt.figure(figsize = (10, 6))

        plt.plot(self.xValues, self.yValues, 'o--')
        plt.xlabel('Tick')
        plt.ylabel('SNR')
        plt.tight_layout()
        plt.title('Grader')

        return plt

def primeTime():
    plt.show()
    disableInteractive()
    input("Press any key to close all figures\n")
    plt.close('all')

class export():
    DATADIRECTORY = './data'

    TOPIC = "topic"
    DESCRIPTION = "description"
    IDENTIFIER = "identifier"
    QCNAME = "qcname"
    QCNOISE = "qcnoise"
    QCCIRCUIT = "qccircuit"
    SIGNAL = "SIGNAL"
    SIGNALPARAM = "signalparam"
    TRANSFORMPARAM = "transformparam"
    PLOTHANDLE = "plothandle"
    PLOTPARAM = "plotparam"
    GRADERX = "graderx"
    GRADERY = "gradery"
    GITHASH = "githash"


    def __init__(self, topic=None, identifier=None) -> None:
        self.details = dict()

        if topic is not None:
            self.setData(self.TOPIC, topic)
        if identifier is not None:
            self.setData(self.IDENTIFIER, identifier)


    def setData(self, dkey, data):
        self.details[dkey] = data

    def setParam(self, dkey, **kwargs):
        self.details[dkey] = dict()

        for key, value in kwargs.items():
            self.details[dkey][key] = value

    def nsa(self, fhandle, **params):
        self.setParam(self, dkey=type(fhandle).__name__, kwargs=params)

        return fhandle(**params)

    def getBasePath(self):
        path = self.DATADIRECTORY + "/" + self.details[self.TOPIC] + "/" + self.details[self.IDENTIFIER]
        return path

    def createTopicOnDemand(self):
        content = os.listdir(self.DATADIRECTORY)

        topic = self.details[self.TOPIC]

        for c in content:
            if c == topic:
                print(f"Topic {topic} already exists in {self.DATADIRECTORY}")
                return

        try:
            os.mkdir(self.DATADIRECTORY+"/"+topic)
            plot(f"Folder {topic} created in {self.DATADIRECTORY}")
        except Exception as e:
            print(e)

    @staticmethod
    def checkWorkingTree():
        repo = git.Repo(path=export.DATADIRECTORY)

        hcommit = repo.head.commit
        d = hcommit.diff(None)
        if len(d) > 0:
            input(f"Working Tree in {export.DATADIRECTORY} is dirty. You might want to commit first. Press any key to continue regardless")


    def getGitCommitId(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.details[self.GITHASH] = sha

    def safePlot(self):
        pltInstance = self.details[self.PLOTHANDLE]

        path = self.getBasePath() + ".png"
        pltInstance.savefig(path)

    def safeDetails(self):
        path = self.getBasePath() + ".p"

        pickle.dump(self.details, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def doExport(self):
        self.createTopicOnDemand()
        self.getGitCommitId()

        self.safePlot()
        self.safeDetails()



