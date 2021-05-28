#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import numpy as np

from transformers import AutoTokenizer

from jetson_voice.nlp import TextClassificationService
from jetson_voice.utils import load_model, normalize_logits
from .nlp_utils import nlp_dynamic_shapes


class TextClassificationEngine(TextClassificationService):
    """
    Text classification model in TensorRT / onnxruntime.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Load an text classification model from ONNX
        """
        super(TextClassificationEngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'text_classification':
            raise ValueError(f"{self.config.model_path} isn't a Text Classification model (type '{self.config.type}'")
            
        # load model
        dynamic_shapes = {'max' : (1, self.config['dataset']['max_seq_length'])}  # (batch_size, sequence_length)
        
        if nlp_dynamic_shapes:
            dynamic_shapes['min'] = (1, 1)
        
        self.model = load_model(self.config, dynamic_shapes)
        
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer']['tokenizer_name'])
        
        
    def __call__(self, query):
        """
        Perform text classification on the input query.
        
        Parameters:
          query (string) -- The text query, for example:
                             'Today was warm, sunny and beautiful out.'

        Returns a dict with the following keys:
             'class' (int) -- the predicted class index
             'label' (string) -- the predicted class label (and if there aren't labels `str(class)`)
             'score' (float) -- the classification probability [0,1]
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
    
        # retrieve the inputs from the encoded tokens
        inputs = {}
        
        for input in self.model.inputs:
            if input.name not in encodings:
                raise ValueError(f"the encoded inputs from the tokenizer doesn't contain '{input.name}'")

            inputs[input.name] = encodings[input.name]
                    
        # run the model
        logits = self.model.execute(inputs)
        logits = normalize_logits(logits)
        preds  = np.argmax(logits, axis=-1)
 
        # tabulate results
        results = []
        
        for query_idx in range(preds.shape[0]):
            results.append({
                'class' : int(preds[query_idx]),
                'label' : str(preds[query_idx]),
                'score' : logits[query_idx][preds[query_idx]]
            })
            
        if len(results) == 1:
            return results[0]
        else:
            return results
        