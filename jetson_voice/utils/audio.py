#!/usr/bin/env python3
# coding: utf-8

import os
import math
import pprint
import logging
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
    if samples.dtype == np.float32:
        return samples
        
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
        
    
def AudioInput(wav=None, mic=None, sample_rate=16000, chunk_size=16000):
    """
    Create an audio input stream from wav file or microphone.
    Either the wav or mic argument needs to be specified.
    
    Parameters:
        wav (string) -- path to .wav file
        mic (int) -- microphone device index
        sample_rate (int) -- the desired sample rate in Hz
        chunk_size (int) -- the number of samples returned per next() iteration
        
    Returns AudioWavStream or AudioMicStream
    """
    if mic is not None and mic != '':
        return AudioMicStream(mic, sample_rate=sample_rate, chunk_size=chunk_size)
    elif wav is not None and wav != '':
        return AudioWavStream(wav, sample_rate=sample_rate, chunk_size=chunk_size)
    else:
        raise ValueError('either wav or mic argument must be specified')
 
 
class AudioWavStream:
    """
    Audio playback stream from .wav file
    """
    def __init__(self, filename, sample_rate, chunk_size):
        self.filename = filename
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
                
        if not os.path.isfile(filename):
            raise IOError(f'could not find file {filename}')
            
        logging.info(f"loading audio '{filename}'")
        
        self.samples, _ = librosa.load(filename, sr=sample_rate, mono=True)
        self.position = 0

    def open(self):
        pass
        
    def close(self):
        pass
        
    def reset(self):
        self.position = 0
        
    def next(self):
        if self.position >= len(self.samples):
            return None
        
        chunk = self.samples[self.position : min(self.position + self.chunk_size, len(self.samples))]
        
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size-len(chunk)), mode='constant')
            
        self.position += self.chunk_size
        return chunk
        
    def __next__(self):
        samples = self.next()
        
        if samples is None:
            raise StopIteration
        else:
            return samples
        
    def __iter__(self):
        self.position = 0
        return self


class AudioMicStream:
    """
    Live audio stream from microphone input device.
    """
    def __init__(self, device, sample_rate, chunk_size):
        self.stream = None
        self.interface = pa.PyAudio()
        
        self.device_info = find_audio_device(device, self.interface)
        self.device_id = self.device_info['index']
        self.device_sample_rate = sample_rate
        self.device_chunk_size = chunk_size
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        print('Audio Input Device:')
        pprint.pprint(self.device_info)
    
    def __del__(self):
        self.close()
        self.interface.terminate()
        
    def open(self):
        if self.stream:
            return
        
        sample_rates = [self.sample_rate, int(self.device_info['defaultSampleRate']), 16000, 22050, 32000, 44100]
        chunk_sizes = []
        
        for sample_rate in sample_rates:
            chunk_sizes.append(int(self.chunk_size * sample_rate / self.sample_rate))
            
        for sample_rate, chunk_size in zip(sample_rates, chunk_sizes):
            try:    
                logging.info(f'trying to open audio input {self.device_id} with sample_rate={sample_rate} chunk_size={chunk_size}')
                
                self.stream = self.interface.open(format=pa.paInt16,
                                channels=1,
                                rate=sample_rate,
                                input=True,
                                input_device_index=self.device_id,
                                frames_per_buffer=chunk_size)
                                
                self.device_sample_rate = sample_rate
                self.device_chunk_size = chunk_size
                
                break
                
            except OSError as err:
                print(err)
                logging.warning(f'failed to open audio input {self.device_id} with sample_rate={sample_rate}')
                self.stream = None
                
        if self.stream is None:
            logging.error(f'failed to open audio input device {self.device_id} with any of these sample rates:')
            logging.error(str(sample_rates))
            raise ValueError(f"audio input device {self.device_id} couldn't be opened or does not support any of the above sample rates")
                      
        print(f"\naudio stream opened on device {self.device_id} ({self.device_info['name']})")
        print("you can begin speaking now... (press Ctrl+C to exit)\n")
            
    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
     
    def reset(self):
        self.close()
        self.open()
        
    def next(self):
        self.open()
            
        samples = self.stream.read(self.device_chunk_size, exception_on_overflow=False)
        samples = np.frombuffer(samples, dtype=np.int16)
        
        if self.sample_rate != self.device_sample_rate:
            samples = audio_to_float(samples)
            samples = librosa.resample(samples, self.device_sample_rate, self.sample_rate)
            
            if len(samples) != self.chunk_size:
                logging.warning(f'resampled input audio has {len(samples)}, but expected {self.chunk_size} samples')
                
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
        

class AudioOutput:
    """
    Audio output stream to a speaker.
    """
    def __init__(self, device, sample_rate, chunk_size=4096):
        self.stream = None
        
        if device is None:
            self.device_id = None
            logging.warning(f"creating pass-through audio output without a device")
            return
            
        self.interface = pa.PyAudio()
        self.device_info = find_audio_device(device, self.interface)
        self.device_id = self.device_info['index']
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.requested_rate = sample_rate
        
        print('Audio Output Device:')
        pprint.pprint(self.device_info)
        
        self.open()
    
    def __del__(self):
        if self.device_id is None:
            return
            
        self.close()
        self.interface.terminate()
        
    def open(self):
        if self.stream or self.device_id is None:
            return
            
        try:
            self.stream = self.interface.open(format=pa.paFloat32,
                            channels=1, rate=self.sample_rate,
                            frames_per_buffer=self.chunk_size,
                            output=True, output_device_index=self.device_id)
        except:
            self.sample_rate = int(self.device_info['defaultSampleRate'])
            logging.error(f"failed to open audio output device with sample_rate={self.requested_rate}, trying again with sample_rate={self.sample_rate}")
            
            self.stream = self.interface.open(format=pa.paFloat32,
                            channels=1, rate=self.sample_rate,
                            frames_per_buffer=self.chunk_size,
                            output=True, output_device_index=self.device_id)
        
        logging.info(f"opened audio output device {self.device_id} ({self.device_info['name']})")
        
    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
       
    def write(self, samples):
        if self.device_id is None:
            return
            
        self.open()
        samples = audio_to_float(samples)
        
        if self.requested_rate != self.sample_rate:
            samples = librosa.resample(samples, self.requested_rate, self.sample_rate)
            #wav = soundfile.SoundFile('data/audio/resample_test.wav', mode='w', samplerate=self.sample_rate, channels=1)
            #wav.write(samples)
            #wav.close()
            
        self.stream.write(samples.tobytes())
        
        
#
# device enumeration
# 
_audio_device_info = None

def _get_audio_devices(audio_interface=None):
    global _audio_device_info
    
    if _audio_device_info:
        return _audio_device_info
        
    if audio_interface:
        interface = audio_interface
    else:
        interface = pa.PyAudio()
        
    info = interface.get_host_api_info_by_index(0)
    numDevices = info.get('deviceCount')
    
    _audio_device_info = []
    
    for i in range(0, numDevices):
        _audio_device_info.append(interface.get_device_info_by_host_api_device_index(0, i))
    
    if not audio_interface:
        interface.terminate()
        
    return _audio_device_info
     
     
def find_audio_device(device, audio_interface=None):
    """
    Find an audio device by it's name or ID number.
    """
    devices = _get_audio_devices(audio_interface)
    
    try:
        device_id = int(device)
    except ValueError:
        if not isinstance(device, str):
            raise ValueError("expected either a string or an int for 'device' parameter")
            
        found = False
        
        for id, dev in enumerate(devices):
            if device.lower() == dev['name'].lower():
                device_id = id
                found = True
                break
                
        if not found:
            raise ValueError(f"could not find audio device with name '{device}'")
            
    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"invalid audio device ID ({device_id})")
        
    return devices[device_id]
                
   
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
            print("Input Device ID {:d} - '{:s}' (inputs={:.0f}) (sample_rate={:.0f})".format(i,
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
            print("Output Device ID {:d} - '{:s}' (outputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxOutputChannels'), dev_info.get('defaultSampleRate')))
                  
    print('')
    
    
def list_audio_devices():
    """
    Print out information about present audio input and output devices.
    """
    list_audio_inputs()
    list_audio_outputs()

              

              