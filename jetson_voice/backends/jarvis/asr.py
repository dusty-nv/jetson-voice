#!/usr/bin/env python3
# coding: utf-8

import os
import grpc
import logging

import jarvis_api.audio_pb2 as ja
import jarvis_api.jarvis_asr_pb2 as jasr
import jarvis_api.jarvis_asr_pb2_grpc as jasr_srv

from jetson_voice import ASRService
from jetson_voice.utils import audio_to_int16

    
class JarvisASRService(ASRService):
    """
    Jarvis streaming ASR.
    """
    def __init__(self, config, *args, **kwargs):
        super(JarvisASRService, self).__init__(config, *args, **kwargs)
        
        self.config.setdefault('server', 'localhost:50051')
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('frame_length', 1.0)
        self.config.setdefault('language_code', 'en-US')
        self.config.setdefault('top_k', 1)
        self.config.setdefault('enable_automatic_punctuation', True)
        
        self.channel = grpc.insecure_channel(self.config.server)
        self.client = jasr_srv.JarvisASRStub(self.channel)
        
        self.recognition_config = jasr.RecognitionConfig(
            encoding = ja.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz = self.config.sample_rate,
            language_code = self.config.language_code,
            max_alternatives = self.config.top_k,
            enable_automatic_punctuation = self.config.enable_automatic_punctuation
        )
        
        print('recognition config', self.recognition_config)
        
        self.streaming_config = jasr.StreamingRecognitionConfig(
            config = self.recognition_config,
            interim_results = True
        )
        
        
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
        samples = audio_to_int16(samples)
        print('samples', samples.shape)
        
        request = jasr.StreamingRecognizeRequest(audio_content=samples.tobytes())
        responses = self.client.StreamingRecognize(iter([request]))
        
        print(responses)
        return responses
        
        
    @property
    def sample_rate(self):
        """
        The sample rate that the model runs at (in Hz)
        Input audio should be resampled to this rate.
        """
        return self.config.sample_rate
    
    @property
    def frame_length(self):
        """
        Duration in seconds per frame / chunk.
        """
        return self.config.frame_length
        
    @property
    def chunk_size(self):
        """
        Number of samples per frame/chunk (equal to frame_length * sample_rate)
        """
        return int(self.frame_length * self.sample_rate)