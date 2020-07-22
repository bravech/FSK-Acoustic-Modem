from config import *
import numpy as np
from scipy.io import wavfile
from scipy.signal import iirfilter
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

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

# Argmax that returns -1 when values are equal

def triple_argmax(x):
    a, b = x
    if a == b:
        return 0 
    if a > b: 
        return 1 
    if a < b:
        return 2 


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

    

if __name__ == "__main__":
    fs, audio_input = wavfile.read('asdf.wav')
    # print(fs)
    # print(audio_input.shape)
    # plt.plot(audio_input)
    # plt.show()
    low_data = butter_bandpass_filter(audio_input, F_0 - BANDWIDTH // 2, F_0 + BANDWIDTH // 2, fs)
    high_data = butter_bandpass_filter(audio_input, F_1 - BANDWIDTH // 2, F_1 + BANDWIDTH // 2, fs)

    low_data = np.array(list(map(lambda x: x if x > 10000 else 0, low_data)))
    high_data = np.array(list(map(lambda x: x if x > 10000 else 0, high_data)))

    combined = np.array([low_data, high_data])
    high_or_low = np.apply_along_axis(triple_argmax, 0, combined)
    
    N=80
    plt.figure(6)
    high_or_low = lfilter(np.ones(N) / N, [1], high_or_low)[N:]
    plt.plot(high_or_low)
    plt.show()

    # Hardcoded cutoff values
    # TODO: figure out programmatic method of determining cutoff values (calibration?)
    LOW_CUTOFF = 0.3
    HIGH_CUTOFF = 0.6
    # None is when no symbol is present
    NONE_CUTOFF = 0.1

    data_diff = []
    is_symbol = False
    for i in range(len(high_or_low)):
        if high_or_low[i] > NONE_CUTOFF:
            if not is_symbol:
                is_symbol = True
                if high_or_low[i] > HIGH_CUTOFF:
                    data_diff.append(1)
                if high_or_low[i] > LOW_CUTOFF:
                    data_diff.append(0)
        else:
            is_symbol = False
    

    data = []
    for i in range(1, len(data_diff)):
        data.append((data_diff[i] ^ data_diff[i-1]) & 1)

    print(data)
    
    plt.figure(2)
    plt.clf()
    plt.plot(low_data, label="low")
    plt.plot(high_data, label="high")
    plt.show()
    print(high_or_low.shape)
    print(high_or_low)
