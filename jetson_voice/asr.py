#!/usr/bin/env python3
# coding: utf-8

import os
import logging

from .utils import ConfigDict


def ASR(config, *args, **kwargs):
    """
    Loads a streaming ASR service
    """
    if isinstance(config, str):
        ext = os.path.splitext(str)[1].lower()
        
        if len(ext) > 0:
            if ext == 'json':
                config = ConfigDict(path=config)
            else:
                raise ValueError(f"config '{config}' has invalid extension '.{ext}'")
        else:
            config = ConfigDict(model_path=config)
            
    elif isinstance(config, dict):
        config = ConfigDict(config)
    else:
        raise ValueError(f"expected string or dict type, instead got {type(config).__name__}")
    
    logging.info('ASR config\n', config)
    
    
class ASRService():
    """
    Streaming ASR service base class.
    """
    def __init__(self, config, *args, **kwargs):
        pass
        
    def __call__(self, samples):
        """
        Transcribe streaming audio samples to text, returning the running phrase.
        Phrases are broken up when a break in the audio is detected (i.e. end of sentence)
        
        Parameters:
          samples (array) -- Numpy array of audio samples.

        Returns a list[dict] of the running transcripts with the following keys:
        
          text (string) -- the transcript of the current sentence
          words (list[dict]) -- a list of word dicts that make up the sentence
          end (bool) -- if true, end-of-sentence due to silence
          
        Each transcript represents one phrase/sentence.  When a sentence has been determined
        to be ended, it will be marked with end=True.  Multiple sentence transcripts can be 
        returned if one just ended and another is beginning. 
        """
        pass