#!/usr/bin/env python3
# coding: utf-8

import sys

from jetson_voice import ASR, AudioStream, ConfigArgParser, list_audio_devices
    
    
parser = ConfigArgParser()

parser.add_argument('--model', default='quartznet', type=str, help='path to model, service name, or json config file')
parser.add_argument('--wav', default='', type=str, help='path to input wav/ogg/flac file')
parser.add_argument('--mic', default=-1, type=int, help='device number of input microphone')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)
    
# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()
    
# load the model
asr = ASR(args.model)

# create the audio stream
stream = AudioStream(wav=args.wav, mic=args.mic, 
                     sample_rate=asr.sample_rate, 
                     chunk_size=asr.chunk_size)

# run transcription
for samples in stream:
    results = asr(samples)
    
    if asr.classification:
        print(f"class '{results[0]}' ({results[1]:.3f})")
    else:
        for transcript in results:
            print(transcript['text'])
            
            if transcript['end']:
                print('')
                
print('\naudio stream closed.')
    