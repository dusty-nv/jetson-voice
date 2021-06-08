#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import librosa
import logging
import datetime

from jetson_voice import TTS, ConfigArgParser
from soundfile import SoundFile

parser = ConfigArgParser()

parser.add_argument('--model', default='fastpitch_hifigan', type=str, help='path to model, service name, or json config file')
parser.add_argument('--config', type=str, required=True, help='path to test config file')
parser.add_argument('--rms-threshold', type=float, default=0.005, help='threshold for comparing actual vs expected RMS')
parser.add_argument('--length-threshold', type=float, default=0.1, help='threshold for comparing actual vs expected audio length (in seconds)')
parser.add_argument('--generate', action='store_true', help='generate the expected outputs')
parser.add_argument("--output-dir", default='', help='output directory to save generated audio')

args = parser.parse_args()

if args.output_dir == '':
    args.output_dir = os.path.join('data/tests/tts', args.model, datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
print(args)

print('')
print('----------------------------------------------------')
print(' RUNNING TEST (TTS)')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print('')

# load test config
with open(args.config) as config_file:
    test_config = json.load(config_file)

# load the model
tts = TTS(args.model)

# list of (passed, num_outputs) tuples
passed = 0

# run tests
for idx, test in enumerate(test_config):
    audio = tts(test['text'])
    
    wav_path = os.path.join(args.output_dir, f"{idx}.wav")
    wav = SoundFile(wav_path, mode='w', samplerate=tts.sample_rate, channels=1)
    wav.write(audio)
    wav.close()
    
    actual_length = len(audio) / tts.sample_rate
    actual_rms = float(librosa.feature.rms(y=audio, frame_length=len(audio), center=False)[0][0])
    
    print(f"'{test['text']}'")
    print(f"audio length = {actual_length}s, RMS = {actual_rms}")
    print(f"saved audio to '{wav_path}'\n")
    
    if 'outputs' not in test:
        test['outputs'] = {}
    
    if args.model not in test['outputs']:
        args.generate = True
        
    if args.generate:
        test['outputs'][args.model] = (actual_length, actual_rms)
    else:
        expected_length, expected_rms = test['outputs'][args.model]
        
        length_diff = abs(expected_length - actual_length)
        rms_diff = abs(expected_rms - actual_rms)
        
        if length_diff > args.length_threshold:
            logging.error(f"failed test - length difference of {length_diff}s exceeded threshold of {args.length_threshold} (actual={actual_length}s, expected={expected_length}s)")
            logging.error(f"              '{test['text']}'")
            continue
            
        if rms_diff > args.rms_threshold:
            logging.error(f"failed test - RMS difference of {rms_diff} exceeded threshold of {args.rms_threshold} (actual={actual_rms}, expected={expected_rms})")
            logging.error(f"              '{test['text']}'")
            continue
        
        passed += 1

if args.generate:
    print('')
    logging.info(f"generated expected outputs, saving to '{args.config}'")
    
    with open(args.config, 'w') as config_file:
        json.dump(test_config, config_file, indent=3)
        
    sys.exit(127)

# test summary
print('')
print('----------------------------------------------------')
print(' TEST RESULTS (TTS)')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print(f'   passed: {passed} / {len(test_config)}')
print('')

if passed != len(test_config):
    logging.error(f"failed test '{args.config}' with model '{args.model}'")
    sys.exit(1)
