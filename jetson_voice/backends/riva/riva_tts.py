#!/usr/bin/env python3
# coding: utf-8

import os
import grpc
import logging
import numpy as np

import riva_api.audio_pb2 as ra
import riva_api.riva_tts_pb2 as rtts
import riva_api.riva_tts_pb2_grpc as rtts_srv

from jetson_voice import TTSService

    
class RivaTTSService(TTSService):
    """
    Riva streaming TTS service.  
    """
    def __init__(self, config, *args, **kwargs):
        """
        Open a streaming channel to the Riva server for TTS.  This establishes a connection over GRPC 
        and sends/recieves the requests and responses.
        """
        super(RivaTTSService, self).__init__(config, *args, **kwargs)
        
        self.config.setdefault('server', 'localhost:50051')
        self.config.setdefault('sample_rate', 22050)        # ignored (will always be 22.05KHz)
        self.config.setdefault('voice_name', 'ljspeech')    # ignored
        self.config.setdefault('language_code', 'en-US')

        logging.info(f'Riva TTS service config:\n{self.config}')
        
        self.channel = grpc.insecure_channel(self.config.server)
        self.client = rtts_srv.RivaSpeechSynthesisStub(self.channel)

    def __call__(self, text):
        """
        Generate audio from text.
        
        Parameters:
          text (string) -- The phrase to convert to audio.

        Returns audio samples in a numpy array.
        """
        req = rtts.SynthesizeSpeechRequest()
        
        req.text = text
        req.language_code = self.config.language_code
        req.sample_rate_hz = self.config.sample_rate
        req.voice_name = self.config.voice_name
        req.encoding = ra.AudioEncoding.LINEAR_PCM

        resp = self.client.Synthesize(req)
        
        samples = np.frombuffer(resp.audio, dtype=np.float32)
        return samples
    
    @property
    def sample_rate(self):
        """
        Get the output sample rate (in Hz)
        """
        return self.config.sample_rate