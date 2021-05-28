#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import numpy as np

from transformers import AutoTokenizer

from jetson_voice.nlp import IntentSlotService
from jetson_voice.utils import load_model, normalize_logits
from .nlp_utils import find_subtokens, nlp_dynamic_shapes


class IntentSlotEngine(IntentSlotService):
    """
    Joint Intent and Slot classification model in TensorRT / onnxruntime.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Load an Intent/Slot classification model from ONNX
        """
        super(IntentSlotEngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'intent_slot':
            raise ValueError(f"{self.config.model_path} isn't an Intent/Slot model (type '{self.config.type}'")
            
        # load model
        dynamic_shapes = {'max' : (1, self.config['language_model']['max_seq_length'])}  # (batch_size, sequence_length)
        
        if nlp_dynamic_shapes:
            dynamic_shapes['min'] = (1, 1)
        
        self.model = load_model(self.config, dynamic_shapes)
        
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer']['tokenizer_name'])
        self.null_slot = self.slot_labels[-1]  # 'O' in assistant dataset - always the last label?
        
        
    def __call__(self, query):
        """
        Perform intent/slot classification on the input query.
        
        Parameters:
          query (string) -- The text query, for example:
                             'What is the weather in San Francisco tomorrow?'

        Returns a dict with the following keys:
             'intent' (string) -- the classified intent label
             'score' (float) -- the intent probability [0,1]
             'slots' (list[dict]) -- a list of dicts, where each dict has the following keys:
                  'slot' (string) -- the slot label
                  'text' (string) -- the slot text from the query
                  'score' (float) -- the slot probability [0,1]
        """
        encodings = self.tokenizer(
            text=query,
            padding='longest' if nlp_dynamic_shapes else 'max_length',
            truncation=True,
            max_length=self.config['language_model']['max_seq_length'],
            return_tensors='np',
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        # during slot classification, we want to ignore slots from subtokens and special tokens 
        subtoken_mask = find_subtokens(encodings, method='subtoken_delimiters')
        ignore_mask = subtoken_mask | encodings['special_tokens_mask']
    
        # retrieve the inputs from the encoded tokens
        inputs = {}
        
        for input in self.model.inputs:
            if input.name not in encodings:
                raise ValueError(f"the encoded inputs from the tokenizer doesn't contain '{input.name}'")

            inputs[input.name] = encodings[input.name]
                    
        # run the model
        intent_logits, slot_logits = self.model.execute(inputs)

        intent_logits = normalize_logits(intent_logits)
        slot_logits = normalize_logits(slot_logits)

        intent_preds = np.argmax(intent_logits, axis=-1)
        slot_preds = np.argmax(slot_logits, axis=-1)

        # convert numerical outputs to intent/slot labels
        results = []

        for query_idx, intent_id in enumerate(intent_preds):
            results.append({
                'intent' : self.intent_label(intent_id),
                'score' : intent_logits[query_idx][intent_id],
                'slots' : []
            })
                
        for query_idx, slots in enumerate(slot_preds):
            query_slots = [self.slot_label(slot) for slot in slots]

            for token_idx, slot in enumerate(query_slots):
                # ignore unclassified slots or masked tokens
                if slot == self.null_slot or ignore_mask[query_idx][token_idx]:
                    continue
                    
                # convert from token index back to the query string
                chars = encodings.token_to_chars(query_idx, token_idx)
                text = query[chars[0]:chars[1]]      # queries[query_idx]
                
                # append subtokens from the query to the text
                for subtoken_idx in range(token_idx+1, len(query_slots)):
                    if subtoken_mask[query_idx][subtoken_idx]:
                        subtoken_chars = encodings.token_to_chars(query_idx, subtoken_idx)
                        text += query[subtoken_chars[0]:subtoken_chars[1]]
                    else:
                        break
                        
                results[query_idx]['slots'].append({
                    'slot' : slot,
                    'text' : text,
                    'score' : slot_logits[query_idx][token_idx][slots[token_idx]]
                })
        
        if len(results) == 1:
            return results[0]
        else:
            return results
            
    @property
    def intent_labels(self):
        """
        List of the intent class labels.
        """
        return self.config['data_desc']['intent_labels']
    
    def intent_label(self, index):
        """
        Return an intent label by index (with bounds checking)
        """
        return self.intent_labels[int(index)] if index < len(self.intent_labels) else 'Unknown_Intent'
        
    @property
    def slot_labels(self):
        """
        List of the slot class labels.
        """
        return self.config['data_desc']['slot_labels']
    
    def slot_label(self, index):
        """
        Return a slot label by index (with bounds checking)
        """
        return self.slot_labels[int(index)] if index < len(self.slot_labels) else self.null_slot
        