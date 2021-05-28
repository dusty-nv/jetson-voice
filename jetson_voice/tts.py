#!/usr/bin/env python3
# coding: utf-8

import os
import time
import logging

from jetson_voice.utils import load_resource


def TTS(resource, *args, **kwargs):
    """
    Loads a TTS service or model.
    See the TTSService class for the signature that implementations use.
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.models.TTSEngine',
        'onnxruntime' : 'jetson_voice.models.TTSEngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs)

    
class TTSService():
    """
    TTS service base class.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Create service instance.
        """
        self.config = config
        
    def __call__(self, text):
        """
        Generate audio from text.
        
        Parameters:
          text (string) -- The phrase to convert to audio.

        Returns audio samples in a numpy array.
        """
        pass
    
    @property
    def sample_rate(self):
        """
        Get the output sample rate (in Hz)
        """
        pass
        
        
if __name__ == "__main__":

    from jetson_voice import list_audio_devices, ConfigArgParser
    from soundfile import SoundFile
    
    import pprint
    import pyaudio
    
    parser = ConfigArgParser()
    
    parser.add_argument('--model', default='fastpitch_hifigan', type=str)
    parser.add_argument('--text', default='Hello, how are you today?', type=str)
    parser.add_argument('--warmup', type=int, default=9, help='the number of warmup runs')
    parser.add_argument("--output-device", type=int, default=None, help='output audio device to use')
    parser.add_argument("--output-wav", type=str, default=None, help='output wav file to write to')
    parser.add_argument('--list-devices', action='store_true', help='list audio input devices')
    
    args = parser.parse_args()
    print(args)
    
    # list audio devices
    if args.list_devices:
        list_audio_devices()
        
    # load the model
    tts = TTS(args.model)
    
     # display the text
    print(f"\n'{args.text}'\n")
    
    # run the TTS
    for run in range(args.warmup+1):
        start = time.perf_counter()
        audio = tts(args.text)
        stop = time.perf_counter()
        latency = stop-start
        duration = audio.shape[0]/tts.sample_rate
        print(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
        
    # output the audio
    if args.output_device is not None:
        p = pyaudio.PyAudio()
        stream = p.open(output_device_index=args.output_device, 
                        format=pyaudio.paFloat32, 
                        channels=1, rate=tts.sample_rate, output=True)
        stream.write(audio.tobytes())
        stream.close_stream()
        stream.close()
        
    if args.output_wav is not None:
        wav = SoundFile(args.output_wav, mode='w', samplerate=tts.sample_rate, channels=1)
        wav.write(audio)
        wav.close()
        print(f"Wrote audio to {args.output_wav}")
    