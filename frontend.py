from matplotlib import colors
import matplotlib as mpl
import numpy as np
from numpy import pi, string_
from scipy import signal as scipySignal
import matplotlib.pyplot as plt
from math import log, floor
from copy import deepcopy
import os
import pickle
import git
from cycler import cycler
# from matplotlib.gridspec import GridSpec

import librosa

from qbstyles import mpl_style

class frontend():
    # COLORMAP = 'gist_ncar'
    COLORMAP = 'twilight'
    SHADING='nearest'
    MAIN='#06574b'
    WHITE='#FFFFFF'
    GRAY='#BBBBBB'
    HIGHLIGHT='#9202e1'
    LIGHTGRAY='#EEEEEE'

    DARK=False
    clickEventHandled = True

    @staticmethod
    def enableInteractive():
        global plt
        plt.ion()

    @staticmethod
    def disableInteractive():
        global plt
        plt.ioff()

    @staticmethod
    def setTheme(dark=DARK):
        frontend.DARK = dark

        mpl_style(dark=frontend.DARK, minor_ticks=False)

    @staticmethod
    def primeTime():
        

        plt.show()
        frontend.disableInteractive()
        input("Press any key to close all figures\n")
        plt.close('all')

    @staticmethod
    def on_click(event):
        '''
        Taken from: https://stackoverflow.com/questions/9012081/matplotlib-grab-single-subplot-from-multiple-subplots
        '''

        if not frontend.clickEventHandled:
            return

        ax = event.inaxes
        if ax is not None:
            # Occurs when a region not in an axis is clicked...
            if int(event.button) == 1:
                # On left click, zoom the selected axes
                ax._orig_position = ax.get_position()
                ax.set_position([0.1, 0.1, 0.85, 0.85])
                for axis in event.canvas.figure.axes:
                    # Hide all the other axes...
                    if axis is not ax:
                        axis.set_visible(False)
                event.canvas.draw()
            elif int(event.button) == 2:
                ax.remove()
                # ax.set_visible(False)
                event.canvas.draw()

                pass
            elif int(event.button) == 3:
                # On right click, restore the axes
                try:
                    ax.set_position(ax._orig_position)
                    for axis in event.canvas.figure.axes:
                        axis.set_visible(True)
                except AttributeError as e:
                    # If we haven't zoomed, ignore...
                    print(e.with_traceback())
                    pass

                event.canvas.draw()

        frontend.clickEventHandled = True

    def _show(self, yData:np.array, x1Data:np.array, title:str, xlabel:str, ylabel:str, x2Data:np.array=None, subplot:tuple=None, plotType:str='stem', log:bool=False):
        # fighandle = plt.figure()
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        mpl.rcParams['axes.prop_cycle'] = cycler('color',[frontend.MAIN, frontend.HIGHLIGHT])

        fig = plt.gcf()
        if subplot is not None:
            plt.subplot(*subplot,frameon=False)
            plt.subplots_adjust(wspace=0.58, top=0.9, left=0.081, bottom=0.16)
            fig.set_size_inches(3*int(subplot[1]),int(subplot[0])*4)
        else:
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)
            plt.subplots_adjust(left=0.15, right=0.95, top=0.92)
            fig.set_size_inches(6,6)
            # plt.figure(figsize = (10, 6))

        fig.canvas.mpl_connect('button_press_event', frontend.on_click)
        plt.tight_layout()
        
        

        if x2Data is None:
            ax = plt.gca()
            if log and plotType != 'box':
                ax.set_yscale('log')
                plt.autoscale(False)
                if type(yData) == np.ndarray:
                    plt.ylim(max(min(yData.min()*0.92,0.1),0.01),1)
                else:
                    plt.ylim(max(min(min(yData)*0.92,0.1),0.01),1)
                plt.xlim(min(x1Data), max(x1Data))

            if plotType == 'stem':
                plt.stem(x1Data, yData, linefmt=frontend.MAIN, markerfmt="C1o")
            elif plotType == 'box':
                ax.boxplot(yData)
            else:
                plt.plot(x1Data, yData, 'o--')

        else:
            # ax = plt.gca()
            m = plt.pcolormesh(x2Data, x1Data, yData, cmap=frontend.COLORMAP, shading=frontend.SHADING,linewidth=0, rasterized=True)
            # ax.set_rasterized(True)

                
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return {'x1Data':x1Data, 'yData':yData, 'x2Data':x2Data, 'subplot':subplot, 'plotType':plotType, 'log':log, 'xlabel':xlabel, 'ylabel':ylabel, 'title':title}

class signal(frontend):
    def __init__(self, samplingRate:int=40, amplification:int=1, duration:int=2, nSamples:int=80, signalType:str='sin', path:str='') -> None:
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
        self.f = None

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

        self.y, _ = librosa.load(path, sr=self.samplingRate, duration=self.duration) * self.amplification


        # mel_feat = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
        # all_wave.append(np.expand_dims(mel_feat, axis=2))
        # all_label.append(label)

    def setSamplingRate(self, samplingRate:int):
        """Sets the sampling rate for the current signal instance

        Args:
            samplingRate ([type]): [description]
        """
        self.samplingRate = samplingRate
        self.samplingInterval = 1/self.samplingRate

    def setNSamples(self, duration:int=2, nSamples:int=80):
        # Either use the duration or the number of samples depending on what's longer
        t_max = max(duration, nSamples*self.samplingInterval)

        # Get the closest min. int which is a power of 2
        nSamples = int(t_max/self.samplingInterval)
        nSamples_log2_min = floor(log(nSamples, 2))

        # Update the number of samples and the duration based on the previous modifications
        self.nSamples = 2**nSamples_log2_min
        self.duration = self.nSamples*self.samplingInterval

        return self.duration

    def addFrequency(self, frequency:float, phase:int=0):
        if frequency > self.samplingRate/2:
            print("WARNING: Nyquist not fulfilled!")
            
        self.frequencies.append(frequency)
        self.phases.append(phase)

    def externalSample(self, y, t, f=None):
        self.y = y
        self.t = t
        self.f = f
        self.setNSamples(0,t.size)
        self.lockSampling=True

    def split(self, nSamplesWindow:int, overlapFactor:float=0, windowType:str=None):
        self.sample()

        if windowType == 'hanning':
            window = np.hanning(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hanning window")
        elif windowType == 'hamming':
            window = np.hamming(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hamming window")
        elif windowType == 'blackman':
            window = np.blackman(nSamplesWindow)
            if overlapFactor!=0.5: print("Suggest an overlap factor of 0.5 in combination with hamming window")
        elif windowType == 'kaiser':
            window = np.kaiser(nSamplesWindow, 8.6-overlapFactor*5.2) #starting from 8.6=blackman over 6=hanning and 5=hamming downtp 0=rect
            print(f"Using {8.6-overlapFactor*5.2} as beta value for window type 'kaiser'")
        
        else:
            window = 1.

        hopSize = np.int32(np.floor(nSamplesWindow * (1-overlapFactor)))
        nParts = np.int32(np.ceil(len(self.y) / np.float32(hopSize)))
        
        y_split_list = list()

        for i in range(0,nParts-1): # -1 because e.g with an overlap of 0.5 we will get 2*N - 1 segments
            currentHop = hopSize * i                        # figure out the current segment offset
            
            segment = self.y[currentHop:currentHop+nSamplesWindow]  # get the current segment
            
            #usefull when splitting and overlapping overshoots the available samples
            if segment.size < window.size:
                segment = self.y[-nSamplesWindow:]

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
    
    def show(self, subplot=None, ignorePhaseShift:bool=False, xlabel:str="Time (s)", ylabel:str="Amplitude", title:str=""):

        if self.signalType=='file':
            minSamples = self.y.size-1 # Use all samples
        else:
            self.sample()   # Only sample if not file, as data is assumed to be loaded
            minF = min(self.frequencies)
            maxP = max(self.phases) if not ignorePhaseShift else 0
            maxT = (1/minF + maxP)*2 if minF != 0 else self.duration
            minSamples = int(maxT*self.samplingRate)
        xData = self.t[:minSamples]
        yData = self.y[:minSamples]

        if title=="":
            title=type(self).__name__

        return self._show(yData, xData, title, xlabel, ylabel, subplot=subplot, plotType="plot")

class transform(frontend):
    def __init__(self, transformation, **kwargs):
        # allow getting called with none to access internal tools
        if transformation == None:
            print("Warning: Transformation called with 'None' parameter. Use with caution!")
            self.transformation = None
        else:
            self.transformation = transformation(**kwargs)

    def forward(self, y, **kwargs):
        y_hat = self.transformation.transform(y, **kwargs)

        # n = np.arange(y_hat.shape[0])
        # F = y_hat.shape[0]/y.samplingRate
        # f = n/F
        f = self.calcFreqArray(y, y_hat)

        if len(y_hat.shape) == 2:
            # n = np.arange(y_hat.shape[1])
            # T = y_hat.shape[1]/y.duration
            # t = n/T
            t = self.calcTimeArray(y, y_hat)

            return y_hat, f, t
        else:
            return y_hat, f

    def backward(self, y_hat, **kwargs):
        y = self.transformation.transformInv(y_hat, **kwargs)

        return y, y_hat.t

    def calcFreqArray(self, y, y_hat):
        n = np.arange(y_hat.shape[0])
        F = y_hat.shape[0]/y.samplingRate
        f = n/F

        return f

    def calcTimeArray(self, y, y_hat):
        n = np.arange(y_hat.shape[1])
        T = y_hat.shape[1]/y.duration
        t = n/T

        return t

    def postProcess(self, y_hat, f, t=None, scale=None, autopower=True, normalize=True, fmin=None, fmax=None, samplingRate=None, nMels=None, fOut=None):
        if autopower:
            y_hat = np.float32(np.abs(y_hat))
            # y_hat = np.abs(y_hat)
            n = y_hat.shape[0]//2
            f = f[:n]
            # y_hat =(y_hat[:n]/n if t is None else y_hat[:n,:]/n) 
            y_hat =(y_hat[:n] if t is None else y_hat[:n,:]) 

        if fmax != None:
            if fmax >= f.max():
                print(f"f_max {fmax} is not lower than the actual max frequency {f.max()}")
            else:
                f_idx = int(np.where(f>fmax)[0][0])
                f = f[:f_idx]
                y_hat = y_hat[:f_idx,:]    


        if scale == 'log':
            y_hat = 20*np.log10(y_hat)
            # plt.yscale('log',base=2)
        elif scale == 'mel':
            fSize = f.size if not autopower else f.size*2

            # apply mel filters with normalization:
            # np.inf -> max()=1
            # 1 -> max()=
            mel_basis = librosa.filters.mel(samplingRate, fSize, n_mels=nMels, fmin=fmin, fmax=fmax, norm=np.inf)

            y_hat = np.dot(mel_basis[:,1:], y_hat)
            f = np.dot(mel_basis[:,1:], f)
            # y_hat = 1127*np.log10(1+y_hat/700) # mel scale formula
            # plt.yscale('log',base=2)
        if normalize:
            y_hat = y_hat*(1/y_hat.max()) if y_hat.max() != 0 else y_hat
            # y_hat = y_hat*(1/sqrt(y_hat.shape[0]))

        if t is None:
            return y_hat, f
        else:
            return y_hat, f, t

    def swapaxes(self, y_hat):
        return np.swapaxes(y_hat, 0, 1)

    def show(self, yData, x1Data, x2Data=None, subplot=None, title="", xlabel='', ylabel=''):
        # fighandle = plt.figure()


        if x2Data is None:
            yData = np.abs(yData)
            if xlabel == "":
                xlabel = 'Frequency (Hz)'
            if ylabel == "":
                ylabel = 'Amplitude (abs)'
        else:
            if xlabel == "":
                xlabel ='Time (s)'
            if ylabel == "":
                ylabel ='Frequency (Hz)'
                
        if title=="":
            title = type(self.transformation).__name__
            
        return self._show(yData, x1Data, title, xlabel, ylabel, x2Data=x2Data, subplot=subplot)


class grader(frontend):
    epsilon=1e-10
    def __init__(self):
        self.yValues = np.array([])
        self.xValues = np.array([])

    def correlate2d(self, a, b):
        y_hat_diff = scipySignal.correlate2d(a, b, mode='same')

        return y_hat_diff

    def calculateNoisePower(self, y, y_ref):
        diff = np.abs(np.power(y,2)-np.power(y_ref,2))

        # snr = np.divide(np.sum(np.abs(y_ref)),np.sum(diff)+self.epsilon)
        # snr = 1-(1/np.sum(np.power(y_ref,2)) * np.sum(diff))
        snr = 1-(1/len(y_ref) * np.sum(diff))

        return snr
        return 10*np.log10(snr)

    def log(self, ylabel, xlabel):
        self.yValues = np.append(self.yValues, [ylabel])
        self.xValues = np.append(self.xValues, [xlabel])    

    def show(self, subplot=None):
        yData = self.yValues
        x1Data = self.xValues
        title = 'Grader'
        xlabel = 'Tick'
        ylabel = 'g'
        x2Data = None

        return self._show(yData, x1Data, title, xlabel, ylabel, x2Data=x2Data, subplot=subplot, plotType='plot', log=True)


class export():
    DATADIRECTORY = './data'

    TOPIC = "topic"
    DESCRIPTION = "description"
    IDENTIFIER = "identifier"
    BACKEND = "backend"
    JOBRESULT = "jobresult"
    FILTERRESULT = "filteresult"
    QCCIRCUIT = "qccircuit"
    SIGNAL = "SIGNAL"
    SIGNALPARAM = "signalparam"
    TRANSFORMPARAM = "transformparam"
    PLOTDATA = "plotdata"
    PLOTPARAM = "plotparam"
    GRADERX = "graderx"
    GRADERY = "gradery"
    GITHASH = "githash"
    GENERICDATA = "gendata"


    def __init__(self, topic=None, identifier=None, dataDir=DATADIRECTORY) -> None:
        self.details = dict()

        if topic is not None:
            self.setData(self.TOPIC, topic)
        if identifier is not None:
            self.setData(self.IDENTIFIER, identifier)

        self.DATADIRECTORY = dataDir


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
            print(f"Folder {topic} created in {self.DATADIRECTORY}")
        except Exception as e:
            print(e)

    @staticmethod
    def checkWorkingTree(dataDir=DATADIRECTORY):
        try:
            repo = git.Repo(path=dataDir)
        except FileNotFoundError:
            print("Invalid directory")
            return
        except git.InvalidGitRepositoryError:
            print("Try to initialize this directory as a git repo first")
            return

        export.DATADIRECTORY=dataDir

        try:
            hcommit = repo.head.commit
        except ValueError:
            print("Try to make a commit in this repository first")
            return

        d = hcommit.diff(None)
        if len(d) > 0:
            input(f"Working Tree in {export.DATADIRECTORY} is dirty. You might want to commit first. Press any key to continue regardless")


    def getGitCommitId(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.details[self.GITHASH] = sha

    def safeDetails(self):
        path = self.getBasePath() + ".p"

        pickle.dump(self.details, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def doExport(self):
        self.createTopicOnDemand()
        self.getGitCommitId()

        self.safeDetails()
# ----------------------------------------------------------
# On-Import region
# ----------------------------------------------------------

frontend.setTheme()