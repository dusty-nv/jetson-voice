#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import nltk
import logging

from jetson_voice import ASR, AudioStream, ConfigArgParser


parser = ConfigArgParser()

parser.add_argument('--model', default='quartznet', type=str, help='path to model, service name, or json config file')
parser.add_argument('--config', type=str, required=True, help='path to test config file')
parser.add_argument('--threshold', type=int, default=0, help='threshold for comparing real vs expected outputs')
parser.add_argument('--generate', action='store_true', help='generate the expected outputs')

args = parser.parse_args()
print(args)


print('')
print('----------------------------------------------------')
print(' RUNNING TEST (ASR)')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print('')

# load test config
with open(args.config) as config_file:
    test_config = json.load(config_file)

# load the model
asr = ASR(args.model)

# run tests
test_failures = []

for test in test_config:
    stream = AudioStream(wav=test['wav'], 
                         sample_rate=asr.sample_rate, 
                         chunk_size=asr.chunk_size)

    transcripts = []
    
    for samples in stream:
        results = asr(samples)
        
        for transcript in results:
            print(transcript['text'])
            
            if transcript['end']:
                print('')
                transcripts.append(transcript['text'])
 
    if 'results' not in test:
        test['results'] = {}
    
    if args.model not in test['results']:
        args.generate = True
        
    if args.generate:
        test['results'][args.model] = transcripts
    else:
        expected_results = test['results'][args.model]
        
        if len(transcripts) != len(expected_results):
            logging.error(f"failed test '{test['wav']}' - got {len(transcripts)} transcripts (expected {len(expected_results)})")
            test_failures.append(len(expected_results))
            continue
        
        failures = 0
        
        for i in range(len(transcripts)):
            similarity = nltk.edit_distance(expected_results[i], transcripts[i])
            
            if similarity > args.threshold:
                logging.error(f"failed test '{test['wav']}' - similarity {similarity} exceeded threshold of {args.threshold}")
                logging.error( "  expected:  '{expected_results[i]}'")
                logging.error( "  actual:    '{transcripts[i]}'")
                failures += 1
                
        test_failures.append(failures)

if args.generate:
    print('')
    logging.info(f"generated expected outputs, saving to '{args.config}'")
    
    with open(args.config, 'w') as config_file:
        json.dump(test_config, config_file, indent=3)
        
    sys.exit(0)

# test summary
passed_tests = 0
passed_transcripts = 0
total_transcripts = 0

for idx, failures in enumerate(test_failures):
    if failures == 0:
        passed_tests += 1
        
    num_transcripts = len(test_config[idx]['results'][args.model])
    total_transcripts += num_transcripts
    passed_transcripts += num_transcripts - failures
    
print('')
print('----------------------------------------------------')
print(' TEST RESULTS (ASR)')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print(f'   passed: {passed_tests} / {len(test_config)} audio files')
print(f'           {passed_transcripts} / {total_transcripts} transcripts')
print('')

if passed_tests != len(test_config):
    logging.error(f"failed test '{args.config}' with model '{args.model}'")
    sys.exit(1)
