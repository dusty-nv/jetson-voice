#!/usr/bin/env python3
# coding: utf-8

import os

from ctc_decoder import CTCDecoder
from ctc_utils import find_silent_intervals, merge_words, transcript_from_words

from ctc_decoders import Scorer
from swig_decoders import BeamDecoder, ctc_beam_search_decoder_ex

from jetson_voice.utils import global_config


class CTCBeamSearchDecoder(CTCDecoder):
    """
    CTC beam search decoder that optionally uses a language model.
    """
    def __init__(self, config, vocab, resource_path=None):
        """
        Create a new CTCBeamSearchDecoder.
        
        See CTCDecoder.from_config() to automatically create
        the correct type of instance dependening on config.
        """
        super().__init__(config, vocab)
        self.config.setdefault('word_threshold', -1000.0)
        self.reset()
        
        self.scorer = None    
        #self.num_cores = max(os.cpu_count(), 1)
        
        # set default config
        # https://github.com/NVIDIA/NeMo/blob/855ce265b80c0dc40f4f06ece76d2c9d6ca1be8d/nemo/collections/asr/modules/beam_search_decoder.py#L21
        self.config.setdefault('language_model', None)
        self.config.setdefault('beam_width', 32)#128)
        self.config.setdefault('alpha', 0.7 if self.language_model else 0.0)
        self.config.setdefault('beta', 0.0)
        self.config.setdefault('cutoff_prob', 1.0)
        self.config.setdefault('cutoff_top_n', 40)
        self.config.setdefault('top_k', 3)
        
        # check for language model file
        if self.language_model:
            if not os.path.isfile(self.language_model):
                self.config['language_model'] = os.path.join(resource_path, self.language_model)
                if not os.path.isfile(self.language_model):
                    raise IOError(f"language model file '{self.language_model}' does not exist")
                    
        logging.info('CTCBeamSearchDecoder')
        logging.info(str(self.config))
        
        # create scorer
        if self.language_model:
            self.scorer = Scorer(self.config['alpha'],
                                 self.config['beta'],
                                 model_path=self.language_model,
                                 vocabulary=self.vocab)
            
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
        results = ctc_beam_search_decoder_ex(
            logits.tolist(), 
            self.vocab,
            self.config['beam_width'], 
            self.config['cutoff_prob'], 
            self.config['cutoff_top_n'], 
            self.config['top_k'],
            self.timestep,
            self.scorer)
        
        
        if global_config.debug:
            print('BeamSearch results', len(results))
            for idx, result in enumerate(results):
                print(f"  beam {idx} [{result.score:.3f}] '{result.text}'")
                for word_idx, word in enumerate(result.words):
                    print(f"    word {word_idx} [{word.start_time}:{word.end_time} {word.score:.3f}] '{word.text}'")
                
        words = [{
            'text' : word.text,
            'score' : word.score,
            'start_time' : word.start_time,
            'end_time' : word.end_time
        } for word in results[0].words]
        
        # merge new words with past words
        self.words = merge_words(self.words, words, self.config['word_threshold'], 'similarity')
        
        # look for silent/EOS intervals
        silent_intervals = find_silent_intervals(logits, len(self.vocab), self.timesteps_silence, self.timestep) 
        
        if global_config.debug: 
            print(f'silent intervals:  {silent_intervals}')

        self.timestep += self.timestep_offset
        
        # split the words at EOS intervals
        if len(silent_intervals) > 0:
            wordlists = self.split_words(silent_intervals, self.words) #self.split_vad_eos(self.words)
            transcripts = []
            
            for idx, wordlist in enumerate(wordlists):
                # ignore blanks (silence after EOS has already occurred)
                if len(wordlist) == 0:
                    continue
                    
                # if there is only one wordlist, then it must be EOS
                # if there are multiple, then the last one is not EOS
                end = (len(wordlists) == 1) or (idx < (len(wordlists) - 1))
                
                if end:
                    wordlist = self.rebase_word_times(wordlist)
                    self.reset()            # TODO reset timesteps counter correctly
                else:
                    self.words = wordlist   
                    
                transcripts.append((wordlist, end))
        else:
            transcripts = [(self.words, False)]

        return [{
            'text' : transcript_from_words(words, scores=global_config.global_config.debug, times=global_config.global_config.debug, end=end),
            'words' : words,
            'end' : end
        } for words, end in transcripts]
        
    def reset(self):
        """
        Reset the CTC decoder state at EOS (end of sentence)
        """
        #self.timestep = 0
        #self.tail_silence = 0
        self.words = []
        
    @property
    def language_model(self):
        return self.config['language_model']
 