#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pprint
import json

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts

from omegaconf import OmegaConf, open_dict


model_types = {
    'asr' : nemo_asr.models.ASRModel,
    'asr_classification' : nemo_asr.models.ASRModel,
    'qa' : nemo_nlp.models.QAModel,
    'intent_slot' : nemo_nlp.models.IntentSlotClassificationModel,
    'text_classification' : nemo_nlp.models.TextClassificationModel,
    'token_classification' : nemo_nlp.models.TokenClassificationModel
}

parser = argparse.ArgumentParser()

parser.add_argument('--type', choices=model_types.keys(), type=str, required=True)
parser.add_argument('--model', type=str, required=True)   # 'QuartzNet15x5Base-En'
parser.add_argument('--output', default='', type=str, required=True)

args = parser.parse_args()

print('nemo version:', nemo.__version__)

# load model depending on extension/type
extension = os.path.splitext(args.model)[1].lower()

if extension == '.nemo':
    model = model_types[args.type].restore_from(args.model)
elif extension == '.ckpt':
    model = model_types[args.type].load_from_checkpoint(args.model)
else: #elif: len(extension) == 0:
    model = model_types[args.type].from_pretrained(model_name=args.model)
#else:
#    raise ValueError(f'model {args.model} has invalid extension {extension}')

# add type string so we can more easily track this later   
with open_dict(model._cfg):
    model._cfg.type = args.type
    model._cfg.model_path = os.path.basename(args.output)
    model._cfg.model_origin = args.model
    
print('')
print('###############################################')
print('## Model Config')
print('###############################################')
pprint.pprint(OmegaConf.to_container(model._cfg))
print('')

base_path = os.path.splitext(args.output)[0]
json_path = base_path + '.json'
yaml_path = base_path + '.yaml'

#with open(yaml_path, 'w') as yaml_file:
#  OmegaConf.save(config=model._cfg, f=yaml_file)
#  print('saved model config to {:s}'.format(yaml_path))
  
with open(json_path, 'w') as json_file:
  json.dump(OmegaConf.to_container(model._cfg), json_file, indent=3)
  print('saved model config to {:s}'.format(json_path))
  
model.export(args.output, verbose=True)

print('\nexported {:s} to {:s}'.format(args.model, args.output))
