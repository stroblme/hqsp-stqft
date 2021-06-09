import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def test_melspectogram(y_signal):
    """Generates a melspectrogram using librosa

    Returns the calculated intermediate step (spectrum, spectrum in mel-scale and spectrum in mel-scale as db).
    Intended to be used as reference.
    F_max set accordingly to Nyquist.
    Will generate a new plot figure.

    Args:
        y_signal

    Returns:
        y_hat
        y_hat_mel
        y_hat_mel_db
    """
    fig, ax = plt.subplots()

    y = y_signal.y
    sr = y_signal.samplingRate

    # Copied from https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

    y_hat = np.abs(librosa.stft(y))**2
    y_hat_mel = librosa.feature.melspectrogram(S=y_hat, sr=sr)
    y_hat_mel_dB = librosa.power_to_db(y_hat_mel, ref=np.max)
    img = librosa.display.specshow(y_hat_mel_dB, x_axis='time',

                            y_axis='mel', sr=sr,

                            fmax=sr/2, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    return y_hat, y_hat_mel, y_hat_mel_dB