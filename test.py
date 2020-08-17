from pyaudio import PyAudio, paFloat32, paContinue
import numpy as np
from time import sleep


f = 880
fs = 48000
samps =  fs
def callback(in_data, frame_count, time_info, flag):
    if flag:
        print("Playback Error: %i" % flag)
    
    played_frames = callback.counter
    callback.counter += frame_count
    # return (np.sin(np.linspace()))
    return (np.sin(2*np.pi*np.arange(fs*1)*f/fs).astype(np.float32).tobytes(), paContinue)
pa = PyAudio()
callback.counter = 0
stream = pa.open(format = paFloat32,
                 channels = 1,
                 rate = 48000,
                 output = True,
                 frames_per_buffer = samps,
                 stream_callback = callback)

while stream.is_active():
    sleep(0.1)

stream.close()
pa.terminate()