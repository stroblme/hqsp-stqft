import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def test_stft(y_signal):
    fig, ax = plt.subplots()

    y = y_signal.y
    sr = y_signal.samplingRate

    # Copied from https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

    y_hat = np.abs(librosa.stft(y))**2
    
    return y_hat, sr

def test_mel(y_hat, sr):
    y_hat_mel = librosa.feature.melspectrogram(S=y_hat, sr=sr)
    
    return y_hat_mel

def test_to_db(y_hat, sr):
    y_hat_dB = librosa.power_to_db(y_hat, ref=np.max)

    return y_hat_dB

def test_melspectogram(y_signal):
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

    y_hat_mel_db = test_to_db(*test_mel(*test_stft(y_signal)))
    test_plot(y_hat_mel_db)

    return y_hat_mel_db

def test_plot(y_hat):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(y_hat, x_axis='time',

                            y_axis='mel', sr=sr,

                            fmax=sr/2, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')