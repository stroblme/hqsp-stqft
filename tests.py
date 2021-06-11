from scipy import signal as scipySignal
from scipy.fft import fftshift
import matplotlib.pyplot as plt 
import numpy as np

def test_stft(y_signal, nSamplesWindow=2**10, overlapFactor=0, windowType=None):
    # f, t, y_hat = scipySignal.stft(y_signal.sample(), y_signal.samplingRate, window=windowType, nperseg=nSamplesWindow)
    f, t, y_hat = scipySignal.stft(y_signal.sample(), y_signal.samplingRate)
    
    return y_hat, f, t

def test_show(y_hat, f, t):
    plt.pcolormesh(t, f, np.abs(y_hat), cmap='cividis', shading='auto')
    plt.xlabel('Time [s]')
    plt.ylabel('Freq [Hz]')

def test_mel(y_hat, sr):
    y_hat_mel = librosa.feature.melspectrogram(S=y_hat, sr=sr)
    
    return y_hat_mel

def test_to_db(y_hat):
    y_hat_dB = librosa.power_to_db(y_hat, ref=np.max)

    return y_hat_dB

def test_melspectrogram(y_signal):
    """Generates a melspectrogram using librosa

    Returns the calculated intermediate step (spectrum, spectrum in mel-scale and spectrum in mel-scale as db).
    Intended to be used as reference.
    F_max set accordingly to Nyquist.
    Will generate a new plot figure.

    Args:
        y_signal

    Returns:
        y_hat_mel_db
    """

    y_hat_mel_db = test_to_db(test_mel(*test_stft(y_signal)))
    test_plot(y_hat_mel_db, y_signal.samplingRate)

    return y_hat_mel_db

def test_plot(y_hat, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(y_hat, x_axis='time',

                            y_axis='linear', sr=sr,

                            fmax=sr/2, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')