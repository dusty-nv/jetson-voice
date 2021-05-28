#!/usr/bin/env python3
# coding: utf-8

import os
import logging

# for some reason if PyCUDA isn't initialized before OnnxRuntime
# and TensorRT is also used, it makes TensorRT error
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import onnxruntime as ort


class OnnxRuntimeModel:
    """
    Base class for OnnxRuntime models.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Load an ONNX Runtime model.
        """
        self.config = config
        
        logging.info(f"loading ONNX model '{self.config.model_path}' with onnxruntime")
        self.model = ort.InferenceSession(config.model_path, providers=['CUDAExecutionProvider'])
        logging.info(f"loaded ONNX model '{self.config.model_path}' with onnxruntime")
        
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        
        for idx, binding in enumerate(self.inputs):
            print('')
            print(f"input {idx} - {binding.name}")
            print(f"   shape: {binding.shape}")
            print(f"   type:  {binding.type}")
            print('')
 
    def execute(self, inputs, return_dict=False, **kwargs):
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
          return_dict (bool) -- If True, the results will be returned in a dict of numpy arrays, where the
                                keys are the names of the output binding names. By default, the results will 
                                be returned in a list of numpy arrays, in the same order as the output bindings.
          
        Returns the model output as a numpy array (if only one output), list[ndarray], or dict[ndarray].
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        
        assert len(inputs) == len(self.inputs)
        
        if isinstance(inputs, (list,tuple)):
            inputs = {self.inputs[i].name : input for i, input in enumerate(inputs)}
        elif not isinstance(inputs, dict):        
            raise ValueError(f"inputs must be a list, tuple, or dict (instead got type '{type(inputs).__name__}')")
            
        outputs = self.model.run(None, inputs)
        
        if return_dict:
            return {self.outputs[i].name : output for i, output in enumerate(outputs)}
            
        if len(outputs) == 1:
            return outputs[0]
        
        return outputs