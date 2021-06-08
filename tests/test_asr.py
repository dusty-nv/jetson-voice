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
parser.add_argument('--threshold', type=int, default=0, help='threshold for comparing actual vs expected outputs')
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

# list of (passed, num_outputs) tuples
test_results = []

# run tests
for test in test_config:
    stream = AudioStream(wav=test['wav'], 
                         sample_rate=asr.sample_rate, 
                         chunk_size=asr.chunk_size)

    outputs = []
    
    for samples in stream:
        output = asr(samples)
        
        if asr.classification:
            print(f"class '{output[0]}' ({output[1]:.3f})")
            outputs.append(output[0])
        else:
            for transcript in output:
                print(transcript['text'])
                
                if transcript['end']:
                    print('')
                    outputs.append(transcript['text'])

    if not asr.classification:
        if not transcript['end']: # pick up the last transcript
            outputs.append(transcript['text'])
            
    if 'outputs' not in test:
        test['outputs'] = {}
    
    if args.model not in test['outputs']:
        args.generate = True
        
    if args.generate:
        test['outputs'][args.model] = outputs
    else:
        expected_outputs = test['outputs'][args.model]
        
        if len(outputs) != len(expected_outputs):
            logging.error(f"failed test '{test['wav']}' - got {len(outputs)} outputs (expected {len(expected_outputs)})")
            test_results.append((0, len(expected_outputs)))
            continue
        
        passed = 0
        
        for i in range(len(expected_outputs)):
            similarity = nltk.edit_distance(expected_outputs[i], outputs[i])
            
            if similarity > args.threshold:
                logging.error(f"failed test '{test['wav']}' - similarity {similarity} exceeded threshold of {args.threshold}")
                logging.error( "  expected:  '{expected_outputs[i]}'")
                logging.error( "  actual:    '{outputs[i]}'")
            else:
                passed += 1
                
        test_results.append((passed, len(expected_outputs)))

if args.generate:
    print('')
    logging.info(f"generated expected outputs, saving to '{args.config}'")
    
    with open(args.config, 'w') as config_file:
        json.dump(test_config, config_file, indent=3)
        
    sys.exit(127)

# test summary
passed_tests = 0
passed_outputs = 0
total_outputs = 0

for passed, num_outputs in test_results:
    if passed == num_outputs:
        passed_tests += 1
        
    passed_outputs += passed
    total_outputs += num_outputs

print('')
print('----------------------------------------------------')
print(' TEST RESULTS (ASR)')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print(f'   passed: {passed_tests} / {len(test_config)} audio files')
print(f'           {passed_outputs} / {total_outputs} outputs')
print('')

if passed_tests != len(test_config):
    logging.error(f"failed test '{args.config}' with model '{args.model}'")
    sys.exit(1)
