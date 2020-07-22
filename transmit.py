from config import *
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def gen_sin(freq, sample_rate, length):
    # samples = np.linspace(0, t, int(sample_rate * ), endpoint=False)
    samples = np.arange(length * sample_rate) / sample_rate

    signal = np.sin(2 * np.pi * freq * samples)
    signal *= 32767

    signal = np.int16(signal)
    return signal


if __name__ == "__main__":

    fs = 96000
    test = np.concatenate((gen_sin(300, fs, 0.05), gen_sin(100, fs, 0.05)))

    data = "01001100"
    data_diff = "0"
    for x in data:
        data_diff += bin(int(data_diff[-1]) ^ int(x))[-1]
    signal = np.array([])

    for bit in data_diff:
        if bit == '0':
            signal = np.concatenate((signal, gen_sin(F_0, fs, DUR_SHORT)))
        if bit == '1':
            # signal += gen_sin(F_1, fs, DUR_SHORT)
            signal = np.concatenate((signal, gen_sin(F_1, fs, DUR_SHORT)))

        zs = np.int16(np.zeros(int(DUR_LONG * fs)))
        signal = np.concatenate((signal, zs))
    plt.plot(signal)
    plt.show()
    print(signal.shape)

    wavfile.write("asdf.wav", fs, signal)
