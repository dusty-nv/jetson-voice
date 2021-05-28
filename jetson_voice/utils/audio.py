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
    if samples.dtype == np.int16:
        return samples
    elif samples.dtype == np.float32:
        return (samples * 32768).astype(np.int16)
    else:
        return samples.astype(np.int16)
        
    
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
    else:
        raise ValueError('either wav or mic argument must be specified')
 
 
class AudioMicStream:
    """
    Live audio stream from microphone input device.
    """
    def __init__(self, device_id, sample_rate, chunk_size):
        self.p = pa.PyAudio()
        self.device_id = device_id
        self.device_info = self.p.get_device_info_by_host_api_device_index(0, device_id)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        
        print(self.device_info)
    
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
     
            print(f"\naudio stream opened on device {self.device_id} ({self.device_info['name']})")
            print("you can begin speaking now...\n")
            
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


#
# device enumeration
# 
_audio_device_info = None

def _get_audio_devices():
    global _audio_device_info
    
    if _audio_device_info:
        return _audio_device_info
        
    p = pa.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numDevices = info.get('deviceCount')
    
    _audio_device_info = []
    
    for i in range(0, numDevices):
        _audio_device_info.append(p.get_device_info_by_host_api_device_index(0, i))
    
    p.terminate()
    return _audio_device_info
    
    
def list_audio_inputs():
    """
    Print out information about present audio input devices.
    """
    devices = _get_audio_devices()

    print('')
    print('----------------------------------------------------')
    print(f" Audio Input Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):    
        if (dev_info.get('maxInputChannels')) > 0:
            print("Input Device ID {:d} - {:s} (inputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxInputChannels'), dev_info.get('defaultSampleRate')))
                 
    print('')
    
    
def list_audio_outputs():
    """
    Print out information about present audio output devices.
    """
    devices = _get_audio_devices()
    
    print('')
    print('----------------------------------------------------')
    print(f" Audio Output Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):  
        if (dev_info.get('maxOutputChannels')) > 0:
            print("Output Device ID {:d} - {:s} (outputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxOutputChannels'), dev_info.get('defaultSampleRate')))
                  
    print('')
    
    
def list_audio_devices():
    """
    Print out information about present audio input and output devices.
    """
    list_audio_inputs()
    list_audio_outputs()

              

              