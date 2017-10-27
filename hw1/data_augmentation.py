import numpy as np
import matplotlib.pyplot as plt

from data_processing import *


# normalize
def normalize(data, axis=0):
    _mean = np.mean(data, axis=axis)
    _std = np.std(data, axis=axis)
    return (data - _mean) / _std


def augment_data_by_VTLP(train_label, mfcc_train_data, fbank_train_data, X_raw, Y_raw, factor):
    print('before augment:', X_raw.shape, Y_raw.shape)
    # augmentation
    for i in range(factor):
        alpha = np.random.uniform(0.9, 1.1)
        aug_fbank, freq = logscale_VTLP(fbank_train_data, alpha=alpha)
        X_aug = np.concatenate((mfcc_train_data, aug_fbank), axis=1)
        X_raw = np.concatenate((X_raw, X_aug), axis=0)
        Y_raw = np.concatenate((Y_raw, train_label), axis=0)
    print('after augment:', X_raw.shape, Y_raw.shape)
    return X_raw, Y_raw


def logscale_VTLP(spec, sr=16000, factor=20., alpha=1.0, f0=0.9, fmax=1):
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(
        list(map(
            lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale
        )))
    scale *= (freqbins - 1) / max(scale)

    newspec = np.zeros([timebins, freqbins])
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if i < 1 or i + 1 >= freqbins:
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if totw[i] > 1e-6:
            freqs[i] /= totw[i]

    return newspec, freqs


if __name__ == '__main__':
    phone_full_dict = load_phone_dict()
    # fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_text(phone_full_dict)
    fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_binary()

    X_raw = mfcc_train_data[:800]
    # X_raw = normalize(X_raw)
    plt.figure('original')
    plt.imshow(X_raw.T, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')

    for i in range(5):
        alpha = np.random.uniform(0.9, 1.1)
        plt.figure('alpha=%f' % alpha)
        X_new, freq = logscale_VTLP(X_raw, alpha=alpha)
        plt.imshow(X_new.T, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
        print(freq)
    plt.show()
