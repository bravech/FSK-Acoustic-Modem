from config import *
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import socket
import argparse
from multiprocessing import Process, Pipe, Queue
from pyaudio import PyAudio, paFloat32, paContinue
from time import sleep

SOUND_FRAME_SIZE = None 
sym_q = Queue()


# Messy for back-to-back signals, since the sine wave starts from 0 every time & doesn't reuse past value as start
def gen_sin(freq, sample_rate, length):
    #thanks stack overflow
    return np.sin(2*np.pi*np.arange(SOUND_FRAME_SIZE)*freq/sample_rate).astype(np.float32).tobytes()

def network_producer(fs, symb_q, s):
    
    
    # Single-threaded server (sound output device only has one "thread")
    s.listen(1)


    # Main "server" loop
    (clientsocket, address) = s.accept()
    try:
        while True:
            chunk = clientsocket.recv(1)
            if not chunk:
                (clientsocket, address) = s.accept()
            # else:
            #     conn.send(chunk)

            # net_data = conn.recv()
            # test = np.concatenate((gen_sin(300, fs, 0.05), gen_sin(100, fs, 0.05)))

            data = ''.join(format(x, "#08b")[2::] for x in chunk) 
            print(chunk)
            data = '0' * (8 - len(data)) + data
            print(data)

            #Differential encoding (so as to avoid timing)
            data_diff = "0"
            for x in data:
                data_diff += bin(int(data_diff[-1]) ^ int(x))[-1]

            # map(symb_q.put, list(data_diff))

            for x in data_diff:
                map(symb_q.put, '2'*9)
                symb_q.put(x)

            # plt.plot(signal)
            # plt.show()

            # samp_queue.put(signal)

            # for i in range(0, signal.shape[0], SOUND_FRAME_SIZE):
            #     output = signal[i:i + SOUND_FRAME_SIZE].astype(np.float32)
            #     if output.shape[0] != SOUND_FRAME_SIZE:
            #         samp_queue.put(np.concatenate([output ,np.zeros(SOUND_FRAME_SIZE - output.shape[0])]))
            #     else:
            #         samp_queue.put(output)
    except:
        clientsocket.close()


def callback(in_data, frame_count, time_info, status):
    # '0' for low, '1' for high, '2' (or anything else) for space
    if status:
        print("Playback Error: %i" % status)
    # print(callback.q.get())
    # print('hi')

    if not callback.q.empty():
        symb = callback.q.get()
        print(repr(symb))
        if symb == '0':
            return (callback.S_0, paContinue) 
        elif symb == '1':
            return (callback.S_1, paContinue) 
    return (np.zeros(SOUND_FRAME_SIZE).astype(np.float32).tobytes(), paContinue)


# Setup
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Transmit binary data over sound device.")
        parser.add_argument("--IP", default="127.0.0.1")
        parser.add_argument("--Port", default="1234")
        parser.add_argument("--samplerate", action='store', default=44800)
        args = parser.parse_args()
        print(args)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        SOUND_FRAME_SIZE = int(args.samplerate * DUR_SHORT)
    
        s.bind((args.IP, int(args.Port)))

        callback.q = sym_q
        callback.S_0 = gen_sin(F_0, args.samplerate, DUR_SHORT)
        callback.S_1 = gen_sin(F_1, args.samplerate, DUR_LONG)


        # sym_q.put((0, 0, 1))
        # sym_q.put('2')
        sym_q.put('0')

        pa = PyAudio()
        stream = pa.open(format=paFloat32, 
                         channels=1, 
                         rate=args.samplerate,
                         output=True,
                         frames_per_buffer=SOUND_FRAME_SIZE,
                         stream_callback=callback)

        network_proc = Process(target=network_producer, args=(args.samplerate, sym_q, s))
        # sound_proc = Process(target=sound_consumer, args=(sound_pipe,args.samplerate, stream))
        network_proc.start()
        # sound_proc.start()
        while stream.is_active():
            # a = input()
            # a = a.strip()
            # for x in a:
            #     sym_q.put(int(x))
            sleep(0.1)
        pa.close(stream)
        network_proc.terminate()
        s.close()
    except KeyboardInterrupt:
        # stream.close()
        pa.close(stream)
        network_proc.terminate()
        s.close()