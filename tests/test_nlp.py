#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import nltk
import pprint
import logging

from jetson_voice import IntentSlot, QuestionAnswer, TextClassification, TokenClassification, ConfigArgParser


types = ['intent_slot', 'qa', 'text_classification', 'token_classification']
parser = ConfigArgParser()

parser.add_argument('--model', default='distilbert_qa_128', type=str, help='path to model, service name, or json config file')
parser.add_argument('--type', type=str, required=True, choices=types, help='the model type')
parser.add_argument('--config', type=str, required=True, help='path to test config file')
parser.add_argument('--threshold', type=int, default=0, help='threshold for comparing actual vs expected outputs')
parser.add_argument('--generate', action='store_true', help='generate the expected outputs')

args = parser.parse_args()
print(args)

print('')
print('----------------------------------------------------')
print(f' RUNNING TEST (NLP {args.type})')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print('')

# load test config
with open(args.config) as config_file:
    test_config = json.load(config_file)

# load the model
if args.type == 'intent_slot':
    model = IntentSlot(args.model)
elif args.type == 'qa':
    model = QuestionAnswer(args.model)
elif args.type == 'text_classification':
    model = TextClassification(args.model)
elif args.type == 'token_classification':
    model = TokenClassification(args.model)
    
# list of (passed, num_outputs) tuples
test_results = []

# run tests
for test in test_config:
    outputs = []
    
    if args.type == 'intent_slot':
        for query in test['queries']:
            results = model(query)
            
            print('')
            print('query:', query, '\n')
            pprint.pprint(results)
            print('')
            
            result_str = results['intent']
            
            for slot in results['slots']:
                result_str += f" {slot['slot']}={slot['text']}"
                
            outputs.append(result_str)
            
    elif args.type == 'qa':
        for question in test['questions']:
            query = {
                'question': question,
                'context': test['context']
            }
            
            answer = model(query, top_k=1)
            
            print('\n')
            print('context:', query['context'])
            print('')
            print('question:', query['question'])
            print('')
            print('answer:', answer['answer'])
            print('score: ', answer['score'])
            
            outputs.append(answer['answer'])
    
    elif args.type == 'text_classification':
        for query in test['queries']:
            results = model(query)
            
            print('')
            print('query:', query, '\n')
            pprint.pprint(results)
            print('')
            
            outputs.append(results['label'])
    
    elif args.type == 'token_classification':
        for query in test['queries']:
            results = model(query)
            result_str = model.tag_string(query, results)
            
            print('')
            print('query:', query, '\n')
            print(model.tag_string(query, results, scores=True))
            print('')
            
            outputs.append(result_str)
            
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
                logging.error(f"failed test - similarity {similarity} exceeded threshold of {args.threshold}")
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
print(f' TEST RESULTS (NLP {args.type})')
print('----------------------------------------------------')
print(f'   model:  {args.model}')
print(f'   config: {args.config}')
print(f'   passed: {passed_tests} / {len(test_config)} tests')
print(f'           {passed_outputs} / {total_outputs} queries')
print('')

if passed_tests != len(test_config):
    logging.error(f"failed test '{args.config}' with model '{args.model}'")
    sys.exit(1)
