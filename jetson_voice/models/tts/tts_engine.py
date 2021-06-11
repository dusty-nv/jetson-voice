#!/usr/bin/env python3
# coding: utf-8

import os
import re
import logging
import inflect

import numpy as np

from jetson_voice.tts import TTSService
from jetson_voice.utils import global_config, load_model, softmax

      
class TTSEngine(TTSService):
    """
    Text-to-speech synthesis.  This is actually a pipeline of two models,
    the generator model (which generates MEL spectrograms from tokens),
    and the vocoder (which outputs audio from MEL spectrograms)
    """
    def __init__(self, config, *args, **kwargs):
        """
        Loads a streaming ASR model from ONNX or serialized TensorRT engine.
        
        Parameters:
          model (string) -- path to ONNX model or serialized TensorRT engine/plan
          config (string) -- path to model configuration json (will be inferred from model path if empty)
        """
        super(TTSEngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'tts':
            raise ValueError(f"{self.config.model_path} isn't a Text-to-Speech model (type '{self.config.type}'")
            
        # load text->MEL generator model
        self.generator = load_model(self.config.generator)
        
        # load MEL->audio vocoder model
        features = self.config.vocoder.features
        
        dynamic_shapes = {
            'min' : (1, features, 1),
            'opt' : (1, features, 160), # ~5-6 words
            'max' : (1, features, 1024) # ~20-30 words?
        }
        
        self.vocoder = load_model(self.config.vocoder, dynamic_shapes=dynamic_shapes)
        
        # create map of symbol->ID embeddings
        self.symbol_to_id = {s: i for i, s in enumerate(self.get_symbols())}
        
        # create operators for num-to-word conversion
        self.number_regex = re.compile(r'\d+(?:,\d+)?')  # https://stackoverflow.com/a/16321189
        self.number_inflect = inflect.engine()
        
    def __call__(self, text):
        """
        Generate audio from text.
        
        Parameters:
          text (string) -- The phrase to convert to audio.

        Returns audio samples in a numpy array.
        """
        text = self.numbers_to_words(text)   # vocab doesn't include numbers, so convert them to words
        
        pad_symbol = ' '
        min_length = 6
        
        if text[-1].isalnum():      # end with punctuation, otherwise audio is cut-off
            text += pad_symbol
          
        if len(text) < min_length:  # WAR for cuDNN error on JetPack <= 4.5.x
            text = text.ljust(min_length, pad_symbol)
            
        # convert chars to symbol embeddings
        encoded_text = [self.symbol_to_id[s] for s in text.lower() if s in self.symbol_to_id]
        encoded_text = np.expand_dims(np.array(encoded_text, dtype=np.int64), axis=0)
        
        # generate MEL spectrogram + audio
        mels = self.generator.execute(encoded_text)[0]
        audio = self.vocoder.execute(mels)

        return audio.squeeze()
     
    def get_symbols(self):
        """
        Return a list of all the accepted character symbols / embeddings
        """
        _arpabet = [
          'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
          'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
          'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
          'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
          'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
          'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
          'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        _arpabet = ['@' + s for s in _arpabet]
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
        return symbols
     
    def numbers_to_words(self, text):
        """
        Convert instances of numbers to words in the text.
        For example:  "The answer is 42" -> "The answer is forty two."
        """
        number_tokens = self.number_regex.findall(text)
        
        for number_token in number_tokens:
            # TODO test/handle floating-point numbers
            word_text = self.number_inflect.number_to_words(number_token)              
            num_begin = text.index(number_token)

            # insert the words back at the old location
            text = text[:num_begin] + word_text + text[num_begin + len(number_token):]
            
        return text
        
    @property
    def sample_rate(self):
        """
        Get the output sample rate (e.g. 22050, 44100, ect)
        """
        return self.config['vocoder']['sample_rate']