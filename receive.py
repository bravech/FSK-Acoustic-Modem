from config import *
import numpy as np
from scipy.io import wavfile
from scipy.signal import iirfilter
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

F_0_RAD = (2 * np.pi) /  F_0 
F_1_RAD = (2 * np.pi) /  F_1 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    fs, audio_input = wavfile.read('asdf.wav')
    print(fs)
    print(audio_input.shape)
    plt.plot(audio_input)
    plt.show()
    low_data = butter_bandpass_filter(audio_input, F_0 - BANDWIDTH // 2, F_0 + BANDWIDTH // 2, fs)
    high_data = butter_bandpass_filter(audio_input, F_1 - BANDWIDTH // 2, F_1 + BANDWIDTH // 2, fs)
    plt.figure(2)
    plt.clf()
    plt.plot(low_data, label="low")
    plt.plot(high_data, label="high")
    plt.show()



