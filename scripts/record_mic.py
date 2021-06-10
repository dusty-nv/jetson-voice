#!/usr/bin/env python3
# coding: utf-8

import sys
import signal
import argparse

from jetson_voice import AudioInput, list_audio_devices
from soundfile import SoundFile


parser = argparse.ArgumentParser()

parser.add_argument('--mic', default=None, type=str, required=True, help='device name or number of input microphone')
parser.add_argument('--output', default=None, type=str, required=True, help='path to output wav/ogg/flac file')
parser.add_argument('--sample-rate', default=16000, type=int, help='sample rate (in Hz)')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)

# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()
    
# setup exit signal handler        
record = True

def signal_handler(sig, frame):
    global record
    record = False
    print('Ctrl+C recieved, exiting...')
    
signal.signal(signal.SIGINT, signal_handler)

# create the output wav
output_wav = SoundFile(args.output, mode='w', samplerate=args.sample_rate, channels=1)

# create the audio device
input_mic = AudioInput(mic=args.mic, sample_rate=args.sample_rate, chunk_size=4096)
        
# loop until user exits
sample_count = 0

while record:
    samples = input_mic.next()
    output_wav.write(samples)
    sample_count += len(samples)

output_wav.close()
print(f"saved {sample_count / args.sample_rate:.2f} seconds of audio to '{args.output}'")
