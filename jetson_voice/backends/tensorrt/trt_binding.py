#!/usr/bin/env python3
# coding: utf-8

import logging
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit


class Binding:
    """
    Represents an input/output tensor to the model.
    """
    def __init__(self, model, index):
        """
        Parameters:
          model (TRTModel) -- parent model instance
          index (int) -- index of the binding in the model
        """
        self.model = model
        self.index = index

        self.name  = model.trt_engine.get_binding_name(index)
        self.shape = tuple(model.trt_engine.get_binding_shape(index))
        self.dtype = model.trt_engine.get_binding_dtype(index)
        self.input = model.trt_engine.binding_is_input(index)
        self.size  = max(trt.volume(self.shape) * self.dtype.itemsize, 0)
        
        self.dynamic = (self.size <= 0)   
        self.profiles = []
            
        if self.input:
            for i in range(model.trt_engine.num_optimization_profiles):
                profile = model.trt_engine.get_profile_shape(i, index)
                self.profiles.append(dict(
                    min = profile[0],
                    opt = profile[1],
                    max = profile[2]))
        
        self.alloc()
          
    def alloc(self, shape=None):
        """
        Allocate memory for the binding. alloc() is called automatically when needed.
        If new shape is provided, it will update the internal state. 
        """
        if shape is not None:
            self.shape = shape
            
        self.size = trt.volume(self.shape) * self.dtype.itemsize
        
        if self.size <= 0:  # dynamic with shape not yet set
            self.host = None
            self.device = None
            return
            
        self.host = None if self.input else cuda.pagelocked_empty(self.shape, dtype=trt.nptype(self.dtype))
        self.device = cuda.mem_alloc(self.size)
        
    def set_shape(self, shape):
        """
        Set the shape of a dynamic input binding.
        """
        if not self.dynamic:
            raise ValueError(f"binding '{self.name}' is not dynamic")
            
        if not self.input:
            raise ValueError(f"binding '{self.name}' is not an input")
            
        # check to see if the shape already matches
        if self.shape == shape:
            logging.debug(f"binding '{self.name}' already has shape {shape}")
            return
            
        logging.debug(f"binding '{self.name}' has new shape {shape}")
        
        # set the new shape
        if not self.model.trt_context.set_binding_shape(self.index, shape):
            raise ValueError(f"failed to set binding '{self.name}' with shape {shape}")
           
        # re-allocate tensor memory
        self.alloc(shape)
    
    def query_shape(self):
        """
        Updates the shape of a dynamic output binding.
        """
        if not self.dynamic:
            return
            
        if self.input:
            raise ValueError(f"binding '{self.name}' is not an output")
        
        # get the new shape
        shape = tuple(self.model.trt_context.get_binding_shape(self.index))
        
        # check to see if the shape already matches
        if self.shape == shape:
            logging.debug(f"binding '{self.name}' already has shape {shape}")
            return
        
        logging.debug(f"binding '{self.name}' has new output shape {shape}")
        
        # re-allocate tensor memory
        self.alloc(shape)
        return shape
        
    def __str__(self):
        return (
            f"binding {self.index} - '{self.name}'\n"
            f"   input:    {self.input}\n"
            f"   shape:    {self.shape}\n"
            f"   dtype:    {self.dtype}\n"
            f"   size:     {self.size}\n"
            f"   dynamic:  {self.dynamic}\n"
            f"   profiles: {self.profiles}\n"
        )