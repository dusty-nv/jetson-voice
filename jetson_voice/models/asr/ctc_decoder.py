#!/usr/bin/env python3
# coding: utf-8

        
class CTCDecoder:
    """
    CTC decoder base class for ASR.
    """    
    @staticmethod
    def from_config(config, vocab, resource_path=None):
        """
        Static factory function to instantiate the correct
        CTC decoder instance type from the config.
        
           config['type'] == 'greedy' -> CTCGreedyDecoder
           config['type'] == 'beamsearch' -> CTCBeamSearchDecoder
        """
        type = config['type'].lower()
        
        if type == 'greedy':
            from .ctc_greedy import CTCGreedyDecoder
            return CTCGreedyDecoder(config, vocab)
        elif type == "beamsearch":
            from .ctc_beamsearch import CTCBeamSearchDecoder
            return CTCBeamSearchDecoder(config, vocab, resource_path)
        else:
            raise ValueError(f"invalid/unrecognized CTC decoder type '{type}'")
            
    def __init__(self, config, vocab):
        """
        See CTCDecoder.from_config() to automatically create
        the correct type of instance dependening on config.
        """
        self.config = config
        self.vocab = vocab
        self.timestep = 0
        
        self.config.setdefault('vad_eos_duration', 0.65)  # max silent time until end-of-sentence
        self.config.setdefault('timestep_offset', 5)      # number of symbols to drop for smooth streaming
        
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
        pass
        
    def reset(self):
        """
        Reset the CTC decoder state at EOS (end of sentence)
        """
        pass

    def set_timestep(self, timestep):
        """
        Set the current timestep.
        """
        self.timestep = timestep
    
    def set_timestep_delta(self, offset):
        """
        Set the number of timesteps per frame.
        """
        self.timestep_delta = offset - self.config['timestep_offset']
        
    def set_timestep_duration(self, duration):
        """
        Set the duration of each timestep, in seconds.
        """
        self.timestep_duration = duration
        self.timesteps_silence = self.config['vad_eos_duration'] / self.timestep_duration
             
    @property
    def type(self):
        """
        Return the CTC decoder type string ('greedy' or 'beamsearch')
        """
        return self.config['type'].lower() 
        
 