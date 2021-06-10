#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import readline

from jetson_voice import TTS, ConfigArgParser, list_audio_devices
from soundfile import SoundFile


parser = ConfigArgParser()

parser.add_argument('--model', default='fastpitch_hifigan', type=str)
parser.add_argument('--warmup', type=int, default=5, help='the number of warmup runs')
parser.add_argument("--output-device", type=int, default=None, help='output audio device to use')
parser.add_argument("--output-wav", type=str, default=None, help='output directory or wav file to write to')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)

# list audio devices
if args.list_devices:
    list_audio_devices()
  
# load the model
tts = TTS(args.model)

# open output audio device
if args.output_device:
    audio_interface = pyaudio.PyAudio()
    audio_device = p.open(output_device_index=args.output_device, 
                          format=pyaudio.paFloat32, 
                          channels=1, rate=tts.sample_rate, output=True)
                  
# create output wav directory
if args.output_wav:
    wav_is_dir = len(os.path.splitext(args.output_wav)[1]) == 0
    wav_count = 0
    if wav_is_dir and not os.path.exists(args.output_wav):
        os.makedirs(args.output_wav)


while True:
    print(f'\nEnter text, or Q to quit:')
    text = input('> ')
    
    if text.upper() == 'Q':
        sys.exit()
    
    print('')
    
    # run the TTS
    for run in range(args.warmup+1):
        start = time.perf_counter()
        audio = tts(text)
        stop = time.perf_counter()
        latency = stop-start
        duration = audio.shape[0]/tts.sample_rate
        print(f"Run {run} -- Time to first audio: {latency:.3f}s. Generated {duration:.2f}s of audio. RTFx={duration/latency:.2f}.")
        
    # output the audio
    if args.output_device:
        output_device.write(audio.tobytes())
    
    if args.output_wav:
        wav_path = os.path.join(args.output_wav, f'{wav_count}.wav') if wav_is_dir else args.output_wav
        wav = SoundFile(wav_path, mode='w', samplerate=tts.sample_rate, channels=1)
        wav.write(audio)
        wav.close()
        wav_count += 1
        print(f"\nWrote audio to {wav_path}")
    
if args.output_device:
    audio_device.close_stream()
    audio_device.close()

    