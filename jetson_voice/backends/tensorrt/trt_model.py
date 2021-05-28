#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import logging
import pprint

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from .trt_builder import build_engine, TRT_LOGGER
from .trt_binding import Binding


class TRTModel:
    """
    Base class for TensorRT models.
    """
    def __init__(self, config, dynamic_shapes=None, *args, **kwargs):
        """
        Load a TensorRT model from ONNX or serialized TensorRT engine.
        
        Parameters:
          config (ConfigDict) -- configuration dict
          dynamic_shapes (dict) -- dynamic shape profiles for min/max/opt
        """
        self.config = config
            
        # determine if the TensorRT engine already exists
        model_root, model_ext = os.path.splitext(self.config.model_path)
        model_ext = model_ext.lower()
        
        if model_ext == '.onnx':
            engine_path = model_root + '.engine'
            if os.path.exists(engine_path):
                logging.info(f'found cached TensorRT engine at {engine_path}')
                self.config.model_path = engine_path
                model_ext = '.engine'
                
        # either build or load TensorRT engine
        if model_ext == '.onnx':
            self.trt_engine = build_engine(self.config, dynamic_shapes=dynamic_shapes)
        elif model_ext == '.engine' or model_ext == '.plan':
            with open(self.config.model_path, 'rb') as f:
                self.trt_runtime = trt.Runtime(TRT_LOGGER)
                self.trt_engine  = self.trt_runtime.deserialize_cuda_engine(f.read())
        else:
            raise ValueError(f"invalid model extension '{model_ext}' (should be .onnx, .engine, or .plan)")
            
        if self.trt_engine is None:
            raise IOError(f'failed to load TensorRT engine from {self.model_path}')
                
        self.trt_context = self.trt_engine.create_execution_context()
        logging.info(f'loaded TensorRT engine from {self.config.model_path}')

        # create a stream in which to copy inputs/outputs and run inference
        self.stream = cuda.Stream()
        
        # enumerate bindings
        self.bindings = []
        self.inputs  = []
        self.outputs = []

        for i in range(len(self.trt_engine)):
            binding = Binding(self, i)
            self.bindings.append(binding)
            
            if binding.input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
        for binding in self.bindings:
            print(f'\n{binding}')

    def execute(self, inputs, sync=True, return_dict=False, **kwargs):
        """
        Run the DNN model in TensorRT.  The inputs are provided as numpy arrays in a list/tuple/dict.
        Note that run() doesn't perform any pre/post-processing - this is typically done in subclasses.
        
        Parameters:
          inputs (array, list[array], dict[array]) -- the network inputs as numpy array(s).
                         If there is only one input, it can be provided as a single numpy array.
                         If there are multiple inputs, they can be provided as numpy arrays in a
                         list, tuple, or dict.  Inputs in lists and tuples are assumed to be in the
                         same order as the input bindings.  Inputs in dicts should have keys with the
                         same names as the input bindings.
          sync (bool) -- If True (default), will wait for the GPU to be done processing before returning.
          return_dict (bool) -- If True, the results will be returned in a dict of numpy arrays, where the
                                keys are the names of the output binding names. By default, the results will 
                                be returned in a list of numpy arrays, in the same order as the output bindings.
          
        Returns the model output as a numpy array (if only one output), list[ndarray], or dict[ndarray].
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        
        assert len(inputs) == len(self.inputs)
        
        # setup inputs + copy to GPU
        def setup_binding(binding, input):
            input = input.astype(trt.nptype(binding.dtype), copy=False)
            if binding.dynamic: 
                binding.set_shape(input.shape)
            cuda.memcpy_htod_async(binding.device, np.ascontiguousarray(input), self.stream)
            
        if isinstance(inputs, (list,tuple)):
            for idx, input in enumerate(inputs):
                setup_binding(self.bindings[idx], input)
        elif isinstance(inputs, dict):        
            for binding_name in inputs:
                setup_binding(self.find_binding(binding_name), inputs[binding_name])
        else:
            raise ValueError(f"inputs must be a list, tuple, or dict (instead got type '{type(inputs).__name__}')")
            
        assert self.trt_context.all_binding_shapes_specified
        assert self.trt_context.all_shape_inputs_specified 
        
        # query new dynamic output shapes
        for output in self.outputs:
            output.query_shape()

        # run inference
        self.trt_context.execute_async_v2(
            bindings=[int(binding.device) for binding in self.bindings], 
            stream_handle=self.stream.handle
        )
          
        # copy outputs to CPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
          
        # wait for completion
        if sync:
            self.stream.synchronize()
            
        # return results
        if return_dict:
            results = {}
            for output in self.outputs:
                results[output.name] = output.host
            return results
        else:
            if len(self.outputs) == 1:
                return self.outputs[0].host
            else:
                return tuple([output.host for output in self.outputs])

    def find_binding(self, name):
        """
        Lookup an input/output binding by name
        """
        for binding in self.bindings:
            if binding.name == name: 
                return binding   
        logging.error(f"couldn't find binding with name '{name}'")
        return None
        
    def set_shape(self, binding, shape):
        """
        Set the shape of a dynamic binding.
        """
        if isinstance(binding, int):
            binding = self.bindings[binding]
        elif isinstance(binding, str):
            binding = self.find_binding(binding)
        elif not isinstance(binding, dict):
            raise ValueError(f'binding must be specified as int, string, or dict (got {type(binding).__name__})')
            
        binding.set_shape(shape)
    
