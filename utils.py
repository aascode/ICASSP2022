import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

'''
    This function is copied from https://github.com/seth814/Audio-Classification/blob/master/clean.py
'''


def envelope(y, rate, threshold=0.0005):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


'''
    This function calculate spectrogram of signal y at given sampling rate
'''


def calc_spectrogram(y, rate, fft_length=256, hop_length=128):
    assert not np.iscomplexobj(y), "Must not pass in complex numbers"
    max_freq = 8000
    eps = 1e-14
    step = 10  # slide window 10ms
    window_length = 20  # milliseconds
    if max_freq > rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate")
    if step > window_length:
        raise ValueError("step size must not be greater than window size")

    hop_length = int(0.001 * step * rate)
    fft_length = int(0.001 * window_length * rate)

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)

    # The scaling below follows the convention of matplotlib.mlab.spectrogram which is the same as matlabs specgram.
    scale = window_norm * rate

    trunc = (len(y) - fft_length) % hop_length
    x = y[:len(y) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == y[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(rate) / fft_length * np.arange(x.shape[0])

    ind = np.where(freqs <= max_freq)[0][-1] + 1
    # return np.transpose(np.log(x[:ind, :] + eps))
    return (np.log(x[:ind, :] + eps)).T
