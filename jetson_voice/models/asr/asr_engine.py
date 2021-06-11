#!/usr/bin/env python3
# coding: utf-8

import os
import time
import pprint
import logging
import importlib

import torch
import numpy as np

from .ctc_decoder import CTCDecoder

from jetson_voice.asr import ASRService
from jetson_voice.utils import audio_to_float, global_config, load_model, softmax

      
class ASREngine(ASRService):
    """
    Streaming ASR (Automatic Speech Recognition) model in TensorRT or onnxruntime.
    This model is primarily designed to be used on a live audio source like a microphone.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Loads a streaming ASR model from ONNX or serialized TensorRT engine.
        
        Parameters:
          model (string) -- path to ONNX model or serialized TensorRT engine/plan
          config (string) -- path to model configuration json (will be inferred from model path if empty)
        """
        super(ASREngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'asr' and self.config.type != 'asr_classification':
            raise ValueError(f"{self.config.model_path} isn't an ASR model (type '{self.config.type}'")

        # set some default config options that are non-standard in nemo
        if 'streaming' not in self.config:
            self.config['streaming'] = {}
        
        self.config['streaming'].setdefault('frame_length', 1.0)     # duration of signal frame, seconds (TODO shorter defaults for VAD/command classifiers)
        self.config['streaming'].setdefault('frame_overlap', 0.5)    # duration of overlap before/after current frame, seconds
        
        # some config changes for streaming
        if not self.classification:
            self.config['preprocessor']['dither'] = 0.0
            self.config['preprocessor']['pad_to'] = 0
        
            if 'ctc_decoder' not in self.config:
                self.config['ctc_decoder'] = {}
                    
            self.config['ctc_decoder'].setdefault('type', 'greedy')        # greedy or beamsearch
            self.config['ctc_decoder'].setdefault('add_punctuation', True) # add period to the end of sentences
        
            if 'add_punctuation' in kwargs:
                self.config['ctc_decoder']['add_punctuation'] = kwargs['add_punctuation']
                logging.info(f"add_punctuation = {kwargs['add_punctuation']}")
                
        if not self.classification and self.config['preprocessor']['features'] == 64:   # TODO normalization coefficients for citrinet (N=80)
            normalization = {}

            normalization['fixed_mean'] = [
                 -14.95827016, -12.71798736, -11.76067913, -10.83311182,
                 -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
                 -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
                 -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
                 -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
                 -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
                 -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
                 -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
                 -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
                 -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
                 -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
                 -10.10687659, -10.14536695, -10.30828702, -10.23542833,
                 -10.88546868, -11.31723646, -11.46087382, -11.54877829,
                 -11.62400934, -11.92190509, -12.14063815, -11.65130117,
                 -11.58308531, -12.22214663, -12.42927197, -12.58039805,
                 -13.10098969, -13.14345864, -13.31835645, -14.47345634]
                 
            normalization['fixed_std'] = [
                 3.81402054, 4.12647781, 4.05007065, 3.87790987,
                 3.74721178, 3.68377423, 3.69344,    3.54001005,
                 3.59530412, 3.63752368, 3.62826417, 3.56488469,
                 3.53740577, 3.68313898, 3.67138151, 3.55707266,
                 3.54919572, 3.55721289, 3.56723346, 3.46029304,
                 3.44119672, 3.49030548, 3.39328435, 3.28244406,
                 3.28001423, 3.26744937, 3.46692348, 3.35378948,
                 2.96330901, 2.97663111, 3.04575148, 2.89717604,
                 2.95659301, 2.90181116, 2.7111687,  2.93041291,
                 2.86647897, 2.73473181, 2.71495654, 2.75543763,
                 2.79174615, 2.96076456, 2.57376336, 2.68789782,
                 2.90930817, 2.90412004, 2.76187531, 2.89905006,
                 2.65896173, 2.81032176, 2.87769857, 2.84665271,
                 2.80863137, 2.80707634, 2.83752184, 3.01914511,
                 2.92046439, 2.78461139, 2.90034605, 2.94599508,
                 2.99099718, 3.0167554,  3.04649716, 2.94116777]
                 
            self.config['preprocessor']['normalize'] = normalization
        
        # create preprocessor instance
        preprocessor_name = self.config['preprocessor']['_target_'].rsplit(".", 1)
        preprocessor_class = getattr(importlib.import_module(preprocessor_name[0]), preprocessor_name[1])
        logging.debug(f'ASR preprocessor - {preprocessor_class}')

        preprocessor_config = self.config['preprocessor'].copy()
        preprocessor_config.pop('_target_')

        self.preprocessor = preprocessor_class(**preprocessor_config)

        # load the model
        features = self.config.preprocessor.n_mels if self.classification else self.config.preprocessor.features
        time_to_fft = self.sample_rate * (1.0 / 160.0)     # rough conversion from samples to MEL spectrogram dims
        
        dynamic_shapes = {
            'min' : (1, features, int(0.1 * time_to_fft)), # minimum plausible frame length
            'opt' : (1, features, int(1.5 * time_to_fft)), # default of .5s overlap factor (1,64,121)
            'max' : (1, features, int(3.0 * time_to_fft))  # enough for 2s overlap factor
        }
        
        self.model = load_model(self.config, dynamic_shapes)
        
        # create CTC decoder
        if not self.classification:
            self.ctc_decoder = CTCDecoder.from_config(self.config['ctc_decoder'],
                                                      self.config['decoder']['vocabulary'],
                                                      os.path.dirname(self.config.model_path))
                                                      
            logging.info(f"CTC decoder type: '{self.ctc_decoder.type}'")
            
        # create streaming buffer
        self.n_frame_len = int(self.frame_length * self.sample_rate)
        self.n_frame_overlap = int(self.frame_overlap * self.sample_rate)
        
        self.buffer_length = self.n_frame_len + self.n_frame_overlap
        self.buffer_duration = self.buffer_length / self.sample_rate
        
        self.buffer = np.zeros(shape=self.buffer_length, dtype=np.float32)  # 2*self.n_frame_overlap
    
        
    def __call__(self, samples):
        """
        Transcribe streaming audio samples to text, returning the running phrase.
        Phrases are broken up when a break in the audio is detected (i.e. end of sentence)
        
        Parameters:
          samples (array) -- Numpy array of audio samples.

        Returns a dict of the running phrase.
          transcript (string) -- the current transcript
          latest (string) -- the latest additions to the transcript
          end (bool) -- if true, end-of-sequence due to silence
        """
        samples = audio_to_float(samples)
        
        if len(samples) < self.n_frame_len:
            samples = np.pad(samples, [0, self.n_frame_len - len(samples)], 'constant')
            
        self.buffer[:self.n_frame_overlap] = self.buffer[-self.n_frame_overlap:]
        self.buffer[self.n_frame_overlap:] = samples
        
        if global_config.profile: preprocess_begin = time.perf_counter()
        
        # apply pre-processing
        preprocessed_signal, _ = self.preprocessor(
            input_signal=torch.as_tensor(self.buffer, dtype=torch.float32).unsqueeze(dim=0), 
            length=torch.as_tensor(self.buffer.size, dtype=torch.int64).unsqueeze(dim=0)
        )

        if global_config.profile:
            logging.info(f'preprocess time: {time.perf_counter() - preprocess_begin}')
            network_begin = time.perf_counter()
        
        # run the asr model
        logits = self.model.execute(torch_to_numpy(preprocessed_signal))
        logits = np.squeeze(logits)
        logits = softmax(logits, axis=-1)

        if global_config.profile: logging.info(f'network time: {time.perf_counter() - network_begin}')
        
        self.timestep_duration = self.buffer_duration / logits.shape[0]
        self.n_timesteps_frame = int(self.frame_length / self.timestep_duration)
        self.n_timesteps_overlap = int(self.frame_overlap / self.timestep_duration)

        if self.classification:
            argmax = np.argmax(logits)
            prob = logits[argmax]
            return (self.config['labels'][argmax], prob)
        else:
            self.ctc_decoder.set_timestep_duration(self.timestep_duration)
            self.ctc_decoder.set_timestep_delta(self.n_timesteps_frame)

            if global_config.profile: ctc_decoder_begin = time.perf_counter()
            transcripts = self.ctc_decoder.decode(logits)
            if global_config.profile: logging.info(f'ctc_decoder time: {time.perf_counter() - ctc_decoder_begin}')
            
            return transcripts

    @property
    def classification(self):
        """
        Returns true if this is an ASR classification model.
        """
        return self.config.type == 'asr_classification'
        
    @property
    def sample_rate(self):
        """
        The sample rate that the model runs at.
        Input audio should be resampled to this rate.
        """
        return self.config['sample_rate'] if self.classification else self.config['preprocessor']['sample_rate']
        
    @property
    def frame_length(self):
        """
        Duration in seconds per frame / chunk.
        """
        return self.config['streaming']['frame_length']
        
    @property
    def frame_overlap(self):
        """
        Duration of overlap in seconds before/after current frame.
        """
        return self.config['streaming']['frame_overlap']
    
    @property
    def chunk_size(self):
        """
        Number of samples per frame/chunk (equal to frame_length * sample_rate)
        """
        return self.n_frame_len


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
                    