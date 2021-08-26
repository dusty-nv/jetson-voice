#!/usr/bin/env python3
# coding: utf-8

import os
import grpc
import queue
import threading
import logging

import riva_api.audio_pb2 as ra
import riva_api.riva_asr_pb2 as rasr
import riva_api.riva_asr_pb2_grpc as rasr_srv

from jetson_voice import ASRService
from jetson_voice.utils import audio_to_int16

    
class RivaASRService(ASRService):
    """
    Riva streaming ASR service.  
    """
    def __init__(self, config, *args, **kwargs):
        """
        Open a streaming channel to the Riva server for ASR.  This establishes a connection over GRPC 
        and sends/recieves the requests and responses asynchronously.  Incoming audio samples get put
        into a request queue that GRPC picks up, and a thread waits on responses to come in.
        """
        super(RivaASRService, self).__init__(config, *args, **kwargs)
        
        self.config.setdefault('server', 'localhost:50051')
        self.config.setdefault('sample_rate', 16000)
        self.config.setdefault('frame_length', 1.0)
        self.config.setdefault('request_timeout', 2.0)      # how long to wait for new audio to come in
        self.config.setdefault('response_timeout', 0.05)    # how long to wait for results from riva
        self.config.setdefault('language_code', 'en-US')
        self.config.setdefault('enable_automatic_punctuation', True)
        self.config.setdefault('top_k', 1)

        logging.info(f'Riva ASR service config:\n{self.config}')
        
        self.channel = grpc.insecure_channel(self.config.server)
        self.client = rasr_srv.RivaSpeechRecognitionStub(self.channel)
        
        self.recognition_config = rasr.RecognitionConfig(
            encoding = ra.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz = self.config.sample_rate,
            language_code = self.config.language_code,
            max_alternatives = self.config.top_k,
            enable_word_time_offsets = True,
            enable_automatic_punctuation = self.config.enable_automatic_punctuation
        )

        self.streaming_config = rasr.StreamingRecognitionConfig(
            config = self.recognition_config,
            interim_results = True
        )
        
        self.request_queue = queue.Queue()
        self.request_queue.put(rasr.StreamingRecognizeRequest(streaming_config=self.streaming_config))
         
        self.responses = self.client.StreamingRecognize(self)
        self.responses_queue = queue.Queue()
        
        self.response_thread = threading.Thread(target=self.recieve_responses)
        self.response_thread.start()

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

        self.request_queue.put(rasr.StreamingRecognizeRequest(audio_content=samples.tobytes()))
        
        transcripts = []
        
        while True:
            try:
                transcripts.append(self.responses_queue.get(block=True, timeout=self.config.response_timeout))
            except queue.Empty:
                break

        return transcripts
 
    def __next__(self):
        """
        Retrieve the next request containing audio samples to send to the Riva server.
        This is implemented using an iterator interface as that is what GRPC expects.
        """
        try:
            request = self.request_queue.get(block=True, timeout=self.config.request_timeout)
            return request
        except queue.Empty:
            logging.debug(f'{self.config.request_timeout} second timeout occurred waiting for audio samples, stopping Riva ASR service')
            raise StopIteration
        
    def recieve_responses(self):
        """
        Wait to recieve responses from the Riva server and parse them.
        """
        logging.debug('starting Riva ASR service response reciever thread')
        
        for response in self.responses:  # this is blocking
            if not response.results:
                continue

            result = response.results[0]

            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript
            text = text.strip()
            
            if len(text) == 0:
                continue
                
            self.responses_queue.put({
                'text' : text,
                'end' : result.is_final
            })

        logging.debug('exiting Riva ASR service response reciever thread')
        
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

