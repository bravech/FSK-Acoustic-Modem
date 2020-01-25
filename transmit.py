from config import *
import numpy as np
from scipy.io import wavfile


def calcRedundantBits(m):
    # Use the formula 2 ^ r >= m + r + 1
    # to calculate the no of redundant bits.
    # Iterate over 0 .. m and return the value
    # that satisfies the equation

    for i in range(m):
        if (2 ** i >= m + i + 1):
            return i


def posRedundantBits(data, r):
    # Redundancy bits are placed at the positions
    # which correspond to the power of 2.
    j = 0
    k = 1
    m = len(data)
    res = ''

    # If position is power of 2 then insert '0'
    # Else append the data
    for i in range(1, m + r + 1):
        if (i == 2 ** j):
            res = res + '0'
            j += 1
        else:
            res = res + data[-1 * k]
            k += 1

    # The result is reversed since positions are
    # counted backwards. (m + r+1 ... 1)
    return res[::-1]


def calcParityBits(arr, r):
    n = len(arr)
    # For finding rth parity bit, iterate over
    # 0 to r - 1
    for i in range(r):
        val = 0
        for j in range(1, n + 1):

            # If position has 1 in ith significant
            # position then Bitwise OR the array value
            # to find parity bit value.
            if (j & (2 ** i) == (2 ** i)):
                val = val ^ int(arr[-1 * j])
            # -1 * j is given since array is reversed

        # String Concatenation
        # (0 to n - 2^r) + parity bit + (n - 2^r + 1 to n)
        arr = arr[:n - (2 ** i)] + str(val) + arr[n - (2 ** i) + 1:]
    return arr


def hamming_encode(tx_data):  # binary string of data
    # Calculate the no of Redundant Bits Required
    m = len(tx_data)
    r = calcRedundantBits(m)
    print(r)

    # Determine the positions of Redundant Bits
    arr = posRedundantBits(tx_data, r)

    # Determine the parity bits
    arr = calcParityBits(arr, r)
    return arr


def gen_parity_bit(tx_data):  # binary string of data
    int_data = list(map(int, tx_data))
    count = sum(int_data)
    par = count % 2
    return str(par) + tx_data


def gen_sin(freq, sample_rate, length):
    # samples = np.linspace(0, t, int(sample_rate * ), endpoint=False)
    samples = np.arange(length * sample_rate) / sample_rate

    signal = np.sin(2 * np.pi * freq * samples)
    signal *= 32767

    signal = np.int16(signal)
    return signal


if __name__ == "__main__":
    fs = 44100

    # f = F_0
    # t = 0.005
    # t = 1

    # samples = np.arange(t * fs)
    # samples = np.linspace(0, t, int(fs * t), endpoint=False)
    #
    # signal = np.sin(2 * np.pi * f * samples)
    # signal *= 32767
    #
    # signal = np.int16(signal)

    data = "00000000001"
    a = hamming_encode("00000000001")
    a = gen_parity_bit(a)

    signal = np.array([])

    for bit in a:
        if bit == '0':
            signal = np.concatenate(signal, gen_sin(F_0, fs, DUR_SHORT), axis=None)
        if bit == '1':
            # signal += gen_sin(F_1, fs, DUR_SHORT)
            signal = np.concatenate(signal, gen_sin(F_1, fs, DUR_SHORT), axis=None)

        zs = np.int16(np.zeros(int(DUR_LONG * fs)))
        signal = np.concatenate(signal, zs, axis=None)


    wavfile.write("asdf.wav", fs, signal)
