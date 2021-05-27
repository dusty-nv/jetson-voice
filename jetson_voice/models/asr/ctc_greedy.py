#!/usr/bin/env python3
# coding: utf-8

import string
import numpy as np

from .ctc_decoder import CTCDecoder
from .ctc_utils import merge_words, transcript_from_words

from jetson_voice.utils import global_config


class CTCGreedyDecoder(CTCDecoder):
    """
    CTC greedy decoder that simply chooses the highest-probability logits.
    """
    def __init__(self, config, vocab):
        """
        Create a new CTCGreedyDecoder.
        TODO document config.
        
        See CTCDecoder.from_config() to automatically create
        the correct type of instance dependening on config.
        """
        super().__init__(config, vocab)
        
        self.config.setdefault('word_threshold', 0.1)
        
        # add blank symbol to vocabulary
        if '_' not in vocab:
            self.vocab = vocab.copy()
            self.vocab.append('_')
            
        self.reset()
        
    def decode(self, logits):
        """
        Decode logits into words, and merge the new words with the
        previous words from the running transcript.
        
        Returns the running transcript as a list of word dictionaries, 
        where each word dict has he following keys:
        
           'text' (str) -- the text of the word
           'score' (float) -- the probability of the word
           'start_time' (int) -- the start time of the word (in timesteps)
           'end_time' (int) -- the end time of the word (in timesteps)
           
        Note that the start/end times are transformed from timestamps into
        seconds by the ASR engine after CTCDecoder.decode() is called.
        """
        text = []
        prob = 1.0
        probs = []
        
        # select the chars with the max probability
        for i in range(logits.shape[0]):
            argmax = np.argmax(logits[i])
            text.append(self.vocab[argmax])
            probs.append(logits[i][argmax])
              
        if global_config.debug:
            print(text)
            
        # get the max number of sequential silent timesteps (continuing from last frame)
        silent_timesteps = self.end_silent_timesteps
        max_silent_timesteps = 0
        
        for i in range(len(text)):
            if text[i] == '_':
                silent_timesteps += 1
            else:
                max_silent_timesteps = max(silent_timesteps, max_silent_timesteps) if i > 0 else 0
                silent_timesteps = 0
        
        if text[-1] == '_':
            self.end_silent_timesteps = silent_timesteps
           
        # merge repeating chars and blank symbols
        _, words = self.merge_chars(text, probs)  #text[:len(text)-self.config['offset']]
        
        # merge new words with past words
        words = merge_words(self.words, words, self.config['word_threshold'], 'overlap')
        
        # increment timestep (after this frame's timestep is done being used, and before a potential EOS reset)
        self.timestep += self.timestep_offset
        
        # check for EOS
        end = False
        
        if silent_timesteps > self.timesteps_silence:
            end = True
            self.reset()
        else:
            self.words = words
            
        return [{
            'text' : transcript_from_words(words, scores=global_config.debug, times=global_config.debug, end=end),
            'words' : words,
            'end' : end
        }]
           
    def merge_chars(self, text, probs):
        """
        Merge repeating chars and blank symbols into words.
        """
        text_merged = ''
        
        word = None
        words = []

        def ispunct(ch):
            return ch in (string.punctuation + ' ')
            
        for i in range(len(text)):
            if text[i] != self.prev_char and text[i] != '_':
                self.prev_char = text[i]
                
                if text[i] != '_':
                    text_merged += text[i]

                    if not ispunct(text[i]):
                        if word is None:
                            word = {
                                'text' : text[i],
                                'score' : probs[i],
                                'start_char' : len(text_merged) - 1,
                                'end_char' : len(text_merged),
                                'start_time' : self.timestep + i,
                                'end_time' : self.timestep + i + 1
                            }
                        else:
                            word['text'] += text[i]
                            word['score'] *= probs[i]
                            word['end_char'] = len(text_merged)
                            word['end_time'] = self.timestep + i + 1
    
                if ispunct(text[i]) and word is not None:
                    words.append(word)
                    word = None
            
        if word is not None:
            words.append(word)
                
        return text_merged, words
        
    def reset(self):
        """
        Reset the CTC decoder state at EOS (end of sentence)
        """
        self.prev_char = ''
        self.end_silent_timesteps = 0
        self.timestep = 0
        self.words = []

 