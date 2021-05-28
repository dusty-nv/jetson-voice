#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import logging
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(config, 
                 output=None, 
                 precision='fp16',
                 batch_size=1,
                 dynamic_shapes=None,
                 workspace=128, 
                 parse_only=False):
    """
    Build TensorRT engine from ONNX model.
    
    Parameters:
      model (string) -- path to ONNX model
      config (string) -- path to model configuration json (will be inferred from model path if empty)
      output (string) -- path to output serialized TensorRT engine (will be inferred from model path if empty)
      precision (string) -- fp32 or fp16 (int8 not currently supported)
      batch_size (int) -- the maximum batch size (default 1)
      dynamic_shape (dict) -- dynamic shape profiles for min/max/opt
      workspace (int) -- builder workspace memory size (in MB)
      parse_only (bool) -- if true, test parsing the model before exiting without building the TensorRT engine
      
    Returns the built TensorRT engine (ICudaEngine)
    """
    # set default output path
    if output is None or output == '':
        output = f'{os.path.splitext(config.model_path)[0]}.engine'

    # create TensorRT resources
    builder = trt.Builder(TRT_LOGGER)
    builder_config = builder.create_builder_config()
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    builder_config.max_workspace_size = workspace * 1 << 20
    
    # set precision
    precision = precision.lower()
    
    if precision == 'fp16':
        builder_config.set_flag(trt.BuilderFlag.FP16)
        logging.info(f'enabled FP16 precision')
    elif precision == 'int8':
        # https://github.com/NVIDIA/TensorRT/blob/d7baf010e4396c87d58e4d8a33052c01c2d89325/demo/BERT/builder.py#L592
        raise NotImplementedError('INT8 support not yet implemented')
        
    # load the model (from ONNX)
    logging.info(f'loading {config.model_path}')
    
    with open(config.model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            logging.error(f'failed to parse ONNX model {config.model_path}')
            for error in range(parser.num_errors): 
                print (parser.get_error(error))
            return None 

    # create dynamic shape profile
    # TODO refactor this to an abstract .get_dynamic_shapes() implementation in each subclass
    # TODO this currently uses same shape for all inputs - allow for different shape profiles
    profile = builder.create_optimization_profile()
    opt_shape = None
    
    """
    if model_type == 'qa' or model_type == 'text_classification' or model_type == 'token_classification':
        min_shape = (1, 1)  # (batch_size, sequence_length)
        max_shape = (batch_size, model_config['dataset']['max_seq_length'])
    elif model_type == 'intent_slot':
        min_shape = (1, 1)  # (batch_size, sequence_length)
        max_shape = (batch_size, model_config['language_model']['max_seq_length'])
    elif model_type == 'asr':
        features = model_config['preprocessor']['features']
        sample_rate = model_config['preprocessor']['sample_rate']
        sample_to_fft = 1.0 / 160.0  # rough conversion from samples to MEL spectrogram dims
        sample_multiplier = sample_rate * sample_to_fft
        
        min_shape = (batch_size, features, int(0.5 * sample_multiplier))  # minimum plausible frame length
        opt_shape = (batch_size, features, int(1.2 * sample_multiplier))  # default of .1s overlap factor (1,64,121)
        max_shape = (batch_size, features, int(3.0 * sample_multiplier))  # enough for 1s overlap factor
    elif model_type == 'asr_classification':
        features = model_config['preprocessor']['n_mels']
        sample_rate = model_config['sample_rate']
        sample_to_fft = 1.0 / 160.0  # rough conversion from samples to MEL spectrogram dims
        sample_multiplier = sample_rate * sample_to_fft
        
        min_shape = (batch_size, features, int(0.5 * sample_multiplier))  # minimum plausible frame length
        opt_shape = (batch_size, features, int(1.2 * sample_multiplier))  # default of .1s overlap factor (1,64,121)
        max_shape = (batch_size, features, int(3.0 * sample_multiplier))  # enough for 1s overlap factor
    elif model_type == 'tts_vocoder':
        min_shape = (batch_size, model_config['features'], 1)
        opt_shape = (batch_size, model_config['features'], 160)  # ~5-6 words
        max_shape = (batch_size, model_config['features'], 512)  # ~15-20 words?
    else:
        raise NotImplementedError(f"model type '{model_type}' is unrecognized or not supported")
    """           
    
    # TODO support different shape profiles for different input tensors
    if dynamic_shapes is not None:        
        if 'min' not in dynamic_shapes:
            dynamic_shapes['min'] = dynamic_shapes['max']
            
        if 'opt' not in dynamic_shapes:
            dynamic_shapes['opt'] = dynamic_shapes['max']
            
        for i in range(network.num_inputs):  # TODO confirm that input is in fact dynamic
            profile.set_shape(network.get_input(i).name, min=dynamic_shapes['min'], opt=dynamic_shapes['opt'], max=dynamic_shapes['max'])

        builder_config.add_optimization_profile(profile)
                    
    def print_summary():
        print('')
        print('----------------------------------------------------')
        print(' BUILDER CONFIGURATION')
        print('----------------------------------------------------')
        print(f'  - model     {config.model_path}')
        print(f'  - config    {config.path}')
        print(f'  - output    {output}')
        print(f'  - type      {config.type}')
        print(f'  - layers    {network.num_layers}')
        print(f'  - inputs    {network.num_inputs}')
        print(f'  - outputs   {network.num_outputs}')
        print(f'  - precision {precision}')
        print(f'  - workspace {workspace}')
        print('')
        
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            
            print(f'  - input {i}:')
            print(f'      - name     {tensor.name}')
            print(f'      - shape    {tensor.shape}')
            print(f'      - dtype    {tensor.dtype}')
            
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            
            print(f'  - output {i}:')
            print(f'      - name     {tensor.name}')
            print(f'      - shape    {tensor.shape}')
            print(f'      - dtype    {tensor.dtype}')
           
    print_summary()
    
    if parse_only:
        return None
    
    # build the engine
    build_start_time = time.time()
    
    engine = builder.build_engine(network, builder_config)
    
    if engine is None:
        raise ValueError(f"failed to build TensorRT engine for '{config.model_path}'")
        
    build_time_elapsed = (time.time() - build_start_time)
    print(f'\nbuilt engine in {build_time_elapsed} seconds')

    print_summary()
    
    # save engine
    print('\nserializing engine...')
    serialized_engine = engine.serialize()
    with open(output, "wb") as engine_file:
        engine_file.write(serialized_engine)
    print(f'saved engine to {output}')
        
    return engine
        

'''
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'], type=str)
    parser.add_argument('--batch-size', default=1, type=int) # max batch size
    parser.add_argument('--workspace', default=utils.DEFAULT_WORKSPACE, type=int)
    parser.add_argument('--parse-only', action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    build_engine(config=args.config,
                 output=args.output,
                 precision=args.precision,
                 batch_size=args.batch_size,
                 workspace=args.workspace,
                 parse_only=args.parse_only)
'''

