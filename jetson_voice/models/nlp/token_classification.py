#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import numpy as np

from transformers import AutoTokenizer

from jetson_voice.nlp import TokenClassificationService
from jetson_voice.utils import load_model, normalize_logits
from .nlp_utils import find_subtokens, nlp_dynamic_shapes


class TokenClassificationEngine(TokenClassificationService):
    """
    Token classification model (aka Named Entity Recognition) in TensorRT / onnxruntime.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Load an token classification model for NER from ONNX
        """
        super(TokenClassificationEngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'token_classification':
            raise ValueError(f"{self.config.model_path} isn't a Token Classification model (type '{self.config.type}'")
            
        # load model
        dynamic_shapes = {'max' : (1, self.config['dataset']['max_seq_length'])}  # (batch_size, sequence_length)
        
        if nlp_dynamic_shapes:
            dynamic_shapes['min'] = (1, 1)
        
        self.model = load_model(self.config, dynamic_shapes)
        
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer']['tokenizer_name'])
        
        
    def __call__(self, query):
        """
        Perform token classification (NER) on the input query and return tagged entities.
        
        Parameters:
          query (string) -- The text query, for example:
                             "Ben is from Chicago, a city in the state of Illinois, US'

        Returns a list[dict] of tagged entities with the following dictionary keys:
             'class' (int) -- the entity class index
             'label' (string) -- the entity class label
             'score' (float) -- the classification probability [0,1]
             'text'  (string) -- the corresponding text from the input query
             'start' (int) -- the starting character index of the text
             'end'   (int) -- the ending character index of the text
        """
        encodings = self.tokenizer(
            text=query,
            padding='longest' if nlp_dynamic_shapes else 'max_length',
            truncation=True,
            max_length=self.config['dataset']['max_seq_length'],
            return_tensors='np',
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
    
        # during token classification, we want to ignore slots from subtokens and special tokens 
        subtoken_mask = find_subtokens(encodings)
        ignore_mask = subtoken_mask | encodings['special_tokens_mask']
        
        # retrieve the inputs from the encoded tokens
        inputs = {}
        
        for input in self.model.inputs:
            if input.name not in encodings:
                raise ValueError(f"the encoded inputs from the tokenizer doesn't contain '{input.name}'")

            inputs[input.name] = encodings[input.name]
                    
        # run the model
        logits = self.model.execute(inputs)
        logits = normalize_logits(logits)
        
        preds = np.argmax(logits, axis=-1)
        probs = np.amax(logits, axis=-1)
        
        # tabulate results
        tags = []
        label_map = {v: k for k, v in self.config['label_ids'].items()}
        num_queries, num_tokens, _ = logits.shape
        
        assert num_queries == 1  # there should only be 1 input query currently
        
        for query_idx in range(num_queries):
            query_tags = []
            
            for token_idx in range(num_tokens):
                label = label_map[preds[query_idx][token_idx]]
                
                # ignore unclassified slots or masked tokens
                if label == self.config['dataset']['pad_label'] or ignore_mask[query_idx][token_idx]:
                    continue

                # convert from token index back to the query string
                chars = encodings.token_to_chars(query_idx, token_idx)
                
                # append subtokens from the query to the text
                for subtoken_idx in range(token_idx+1, num_tokens):
                    if subtoken_mask[query_idx][subtoken_idx]:
                        chars = (chars[0], encodings.token_to_chars(query_idx, subtoken_idx)[1])
                    else:
                        break

                text = query[chars[0]:chars[1]] # queries[query_idx]

                # strip out punctuation to attach the entity tag to the word not to a punctuation mark
                if not text[-1].isalpha():
                    text = text[:-1]
                    chars = (chars[0], chars[1]-1)
                        
                query_tags.append({
                    'label' : label,
                    'class' : preds[query_idx][token_idx],
                    'score' : probs[query_idx][token_idx],
                    'text' : text,
                    'start' : chars[0],
                    'end' : chars[1]
                })
                
            tags.append(query_tags)
            
        if len(tags) == 1:
            return tags[0]
        else:
            return tags
        