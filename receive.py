from config import *
import numpy as np
from scipy.io import wavfile
from scipy.signal import iirfilter
import matplotlib.pyplot as plt

F_0_RAD = (2 * np.pi) /  F_0 
F_1_RAD = (2 * np.pi) /  F_1 

if __name__ == "__main__":
    fs, audio_input = wavfile.read('asdf.wav')
    print(audio_input.shape)
    plt.plot(audio_input)
    plt.show()
    goertzel_intermediate = np.zeros(audio_input.shape[0])
    precomp = 2 * np.cos(F_0_RAD)
    for i in range(audio_input.shape[0]):
        goertzel_intermediate[i] = audio_input[i] + precomp * (goertzel_intermediate[i-1] if i > 1 else 0) - (goertzel_intermediate[i-2] if i > 2 else 0)
    goertzel_output = np.zeros(audio_input.shape[0])
    for i in range(audio_input.shape[0]):
        goertzel_output[i] = goertzel_intermediate[i] - np.exp

    