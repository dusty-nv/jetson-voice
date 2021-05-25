#!/usr/bin/env python3
# coding: utf-8

import os
import math
import librosa
import soundfile

import pyaudio as pa
import numpy as np


def audio_db(samples):
    """
    Compute RMS of audio samples in dB.
    """
    rms = librosa.feature.rms(y=samples, frame_length=samples.shape[0], center=False)[0][0]

    if rms != 0.0:
        return 20.0 * math.log10(rms)
    else:
        return -100.0
        
        
def audio_to_float(samples):
    """
    Convert audio samples to 32-bit float in the range [-1,1]
    """
    return samples.astype(np.float32) / 32768
  

def audio_to_int16(samples):
    """
    Convert audio samples to 16-bit float in the range [-32767,32767]
    """
    return (samples * 32768).astype(np.int16)
    
    
def AudioStream(wav=None, mic=None, sample_rate=16000, chunk_size=16000):
    """
    Create an audio stream from wav file or microphone.
    Either the wav or mic argument needs to be specified.
    
    Parameters:
        wav (string) -- path to .wav file
        mic (int) -- microphone device index
        sample_rate (int) -- the desired sample rate in Hz
        chunk_size (int) -- the number of samples returned per next() iteration
        
    Returns AudioWavStream or AudioMicStream
    """
    if mic is not None and mic >= 0:
        return AudioMicStream(mic, sample_rate=sample_rate, chunk_size=chunk_size)
    elif wav is not None and wav != '':
        return AudioWavStream(wav, sample_rate=sample_rate, chunk_size=chunk_size)
 
 
class AudioMicStream:
    """
    Live audio stream from microphone input device.
    """
    def __init__(self, device_id, sample_rate, chunk_size):
        self.p = pa.PyAudio()
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        
        print(self.p.get_device_info_by_host_api_device_index(0, device_id))
    
    def __del__(self):
        self.close()
        self.p.terminate()
        
    def open(self):
        if self.stream is None:
            self.stream = self.p.open(format=pa.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=self.device_id,
                            frames_per_buffer=self.chunk_size)
     
    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
       
    def next(self):
        if self.stream is None:
            return None
            
        samples = self.stream.read(self.chunk_size, exception_on_overflow=False)
        samples = np.frombuffer(samples, dtype=np.int16)
        
        return samples
        
    def __next__(self):
        samples = self.next()
        
        if samples is None:
            raise StopIteration
        else:
            return samples
        
    def __iter__(self):
        self.open()
        return self
        

class AudioWavStream:
    """
    Audio playback stream from .wav file
    """
    def __init__(self, filename, sample_rate, chunk_size):
        self.filename = filename
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sf = None
        
        if not os.path.isfile(self.filename):
            raise IOError(f'could not find file {self.filename}')
        
    def __del__(self):
        self.close()
        
    def open(self):
        if self.sf is not None:
            return

        self.sf = soundfile.SoundFile(self.filename, 'rb')
        
        print(self.sf)
        print('  - length', self.sf.frames)
        print('  - format_info', self.sf.format_info)
        print('  - sample_rate', self.sf.samplerate)
        
        if self.sf.samplerate != self.sample_rate:
            raise ValueError(f"'{self.filename}' has a sample rate of {self.sf.samplerate}, but needed {self.sample_rate}")
            
        dtype_options = {'PCM_16': 'int16', 'PCM_32': 'int32', 'FLOAT': 'float32'}
        dtype_file = self.sf.subtype
        
        if dtype_file in dtype_options:
            self.dtype = dtype_options[dtype_file]
        else:
            self.dtype = 'float32'
            
    def close(self):
        if self.sf is not None:
            self.sf.close()
            self.sf = None
    
    def next(self):
        if self.sf is None:
            return None
            
        samples = self.sf.read(self.chunk_size, dtype=self.dtype)

        # pad to chunk size
        if len(samples) < self.chunk_size:
            samples = np.pad(samples, (0, self.chunk_size-len(samples)), mode='constant')
            self.close()

        return samples
        
    def __next__(self):
        samples = self.next()
        
        if samples is None:
            raise StopIteration
        else:
            return samples
        
    def __iter__(self):
        self.open()
        return self
        