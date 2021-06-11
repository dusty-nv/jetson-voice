#!/usr/bin/env python3
# coding: utf-8

import sys
import pprint
import readline

from jetson_voice import NLP, ConfigArgParser


parser = ConfigArgParser()
parser.add_argument('--model', default='distilbert_sentiment', type=str)
args = parser.parse_args()
print(args)

# load the model
model = NLP(args.model)

# QA models should run the nlp_qa.py example
type = model.config.type

if type == 'qa':
    raise ValueError("please run Question/Answer models with the nlp_qa.py sample")


while True:
    print(f'\nEnter {type} query, or Q to quit:')
    query = input('> ')
    
    if query.upper() == 'Q':
        sys.exit()
    
    print('')
    
    results = model(query)
        
    if type == 'intent_slot' or type == 'text_classification':
        pprint.pprint(results)
    
    elif type == 'token_classification':
        print(f'{model.tag_string(query, results, scores=True)}')
        