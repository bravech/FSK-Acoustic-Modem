from config import *
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import socket
import argparse
from multiprocessing import Process, Pipe

def gen_sin(freq, sample_rate, length):
    # samples = np.linspace(0, t, int(sample_rate * ), endpoint=False)
    samples = np.arange(length * sample_rate) / sample_rate

    signal = np.sin(2 * np.pi * freq * samples)
    signal *= 32767

    signal = np.int16(signal)
    return signal

def network_producer(conn):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # s.bind(('127.0.0.1', 1234))
    s.bind((args.IP, int(args.Port)))
    # Single-threaded server (sound output device only has one "thread")
    s.listen(1)


    # Main "server" loop
    (clientsocket, address) = s.accept()

    while True:
        chunk = clientsocket.recv(5)
        if not chunk:
            (clientsocket, address) = s.accept()
        else:
            conn.send(chunk)

def sound_consumer(conn, fs):
    while True:
        net_data = conn.recv()
        # test = np.concatenate((gen_sin(300, fs, 0.05), gen_sin(100, fs, 0.05)))

        data = ''.join(format(x, "#08b") for x in net_data) 
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
        pass

if __name__ == "__main__":
    # Backend setup
    parser = argparse.ArgumentParser(description="Transmit binary data over sound device.")
    parser.add_argument("IP")
    parser.add_argument("Port")
    parser.add_argument("--samplerate", action='store', default=96000)
    args = parser.parse_args()
    print(args)

    net_pipe, sound_pipe = Pipe()
    network_proc = Process(target=network_producer, args=(net_pipe,))
    sound_proc = Process(target=sound_consumer, args=(sound_pipe,args.samplerate))
    network_proc.start()
    sound_proc.start()



