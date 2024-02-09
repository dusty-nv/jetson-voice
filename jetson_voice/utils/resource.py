#!/usr/bin/env python3
# coding: utf-8

import os
import json
import time
import tqdm
import pprint
import logging
import tarfile
import urllib.request
import importlib

from .config import global_config, ConfigDict


def find_resource(path):
    """
    Find a resource by checking some common paths.
    """
    if os.path.exists(path):
        return path
        
    search_dirs = [global_config.model_dir,
                   os.path.join(global_config.model_dir, 'asr'),
                   os.path.join(global_config.model_dir, 'nlp'),
                   os.path.join(global_config.model_dir, 'tts')]
    
    for search_dir in search_dirs:
        search_path = os.path.join(search_dir, path)
        
        if os.path.exists(search_path):
            return search_path
    
    raise IOError(f"failed to locate resource '{path}'")


def load_resource(resource, factory_map, *args, **kwargs):
    """
    Load an instance of a resource from a config or service name.
    The factory_map dict maps the backend names to class names.
    Returns the resource instance, or the config if factory_map is null.
    """
    if isinstance(resource, str):
        root, ext = os.path.splitext(resource)
        
        if len(ext) > 0:
            ext = ext.lower()
            
            if ext == '.json':
                config = ConfigDict(path=resource)
            elif ext == '.onnx' or ext == '.engine' or ext == '.plan':
                config = ConfigDict(path=root + '.json')
            else:
                raise ValueError(f"resource '{resource}' has invalid extension '{ext}'")
        else:
            manifest = download_model(resource)

            if manifest['type'] == 'model':
                config = ConfigDict(path=get_model_config_path(manifest=manifest))
            else:
                config = ConfigDict(backend=manifest['backend'], type=manifest['name'])
    
    elif isinstance(resource, ConfigDict):
        config = resource
    elif isinstance(resource, dict):
        config = ConfigDict(resource)
    else:
        raise ValueError(f"expected string or dict type, instead got {type(resource).__name__}")
    
    config.setdefault('backend', global_config.default_backend)
    
    if factory_map is None:
        return config
        
    if config.backend not in factory_map:
        raise ValueError(f"'{config.path}' has invalid backend '{config.backend}' (valid options are: {', '.join(factory_map.keys())})")
        
    class_name = factory_map[config.backend].rsplit(".", 1)
    class_type = getattr(importlib.import_module(class_name[0]), class_name[1])
    
    logging.debug(f"creating instance of {factory_map[config.backend]} for '{config.path}' (backend {config.backend})")
    logging.debug(class_type)
    
    return class_type(config, *args, **kwargs)
    
    
def load_model(config, dynamic_shapes=None):
    """
    Loads an ONNX model through a backend (either TensorRT or onnxruntime)
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.backends.tensorrt.TRTModel',
        'onnxruntime' : 'jetson_voice.backends.onnxruntime.OnnxRuntimeModel'
    }
    
    config.setdefault('backend', global_config.default_backend)
    config.setdefault('model_path', os.path.splitext(config.path)[0] + '.onnx')
    
    if not os.path.exists(config.model_path):
        model_path = os.path.join(os.path.dirname(config.path), config.model_path)
        
        if not os.path.exists(model_path):
            raise IOError(f"couldn't find file '{config.model_path}'")
        else:
            config.model_path = model_path

    if config.backend not in factory_map:
        raise ValueError(f"'{config.path}' has invalid backend '{config.backend}' (valid options are: {', '.join(factory_map.keys())})")
        
    class_name = factory_map[config.backend].rsplit(".", 1)
    class_type = getattr(importlib.import_module(class_name[0]), class_name[1])
    
    logging.info(f"loading model '{config.model_path}' with {factory_map[config.backend]}")
    logging.debug(class_type)
    
    return class_type(config, dynamic_shapes=dynamic_shapes)
    
    
def load_models_manifest(path=None):
    """
    Load the models manifest file.
    If the path isn't overriden, it will use the default 'data/networks/manifest.json'
    """
    if path is None:
        path = global_config.model_manifest
        
    with open(path) as file:
        manifest = json.load(file)
        
    for key in manifest:
        manifest[key].setdefault('name', key)
        manifest[key].setdefault('config', key + '.json')
        manifest[key].setdefault('type', 'model')
        
    return manifest
    
  
def find_model_manifest(name):
    """
    Find a model manifest entry by name / alias.
    """
    manifest = load_models_manifest()
    
    for key in manifest:
        if key.lower() == name.lower():
            return manifest[key]
        
        if 'alias' in manifest[key]:
            if isinstance(manifest[key]['alias'], str):
                aliases = [manifest[key]['alias']]
            else:
                aliases = manifest[key]['alias']
                
            for alias in aliases:
                if alias.lower() == name.lower():
                    return manifest[key]
      
    raise ValueError(f"could not find '{name}' in manifest '{global_config.model_manifest}'")
    
 
def download_model(name, max_attempts=10, retry_time=5):
    """
    Download a model if it hasn't already been downloaded.
    """
    manifest = find_model_manifest(name)
    
    if manifest is None:
        return None
      
    if manifest['type'] != 'model':
        return manifest
        
    if os.path.exists(get_model_config_path(manifest=manifest)):
        return manifest

    class DownloadProgressBar(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    def attempt_download(attempt):
        logging.info(f"downloading '{manifest['name']}' from {manifest['url']} (attempt {attempt} of {max_attempts})")

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=manifest['name']) as t:
            try:
                filename, _ = urllib.request.urlretrieve(manifest['url'], reporthook=t.update_to)
            except Exception as error:
                t.close()
                logging.error(error)
                return None
                
            return filename
        
    for attempt in range(1, max_attempts+1):
        filename = attempt_download(attempt)
        
        if filename is not None:
            break
            
        logging.error(f"failed to download '{manifest['name']}' from {manifest['url']} (attempt {attempt} of {max_attempts})")
        
        if attempt == max_attempts:
            raise ValueError(f"failed to download '{manifest['name']}' from {manifest['url']} (max attempts exceeded)")
            
        logging.info(f"waiting {retry_time} seconds before trying again...")
        time.sleep(retry_time)
        
    logging.info(f"extracting {filename} to {os.path.join(global_config.model_dir, manifest['domain'], manifest['name'])}")
    
    with tarfile.open(filename, "r:gz") as tar:
        tar.list()
        tar.extractall(path=os.path.join(global_config.model_dir, manifest['domain']))

    os.remove(filename)
    return manifest
        
    
def get_model_config_path(name=None, manifest=None):
    """
    Gets the path to the model config from it's name or manifest entry.
    """
    if name is None and manifest is None:
        raise ValueError('must specify either name or manifest arguments')
        
    if manifest is None:
        manifest = find_model_manifest(name)
        
    if manifest['type'] != 'model':
        raise ValueError(f"resource '{manifest['name']}' is not a model (type='{manifest['type']}')")
    
    if len(os.path.dirname(manifest['config'])) > 0:  # if full path is specified
        return os.path.join(global_config.model_dir, manifest['domain'], manifest['config'])
    else:  
        return os.path.join(global_config.model_dir, manifest['domain'], manifest['name'], manifest['config'])
    
   
def list_models():
    """
    Print out the models available.
    """
    manifest = load_models_manifest()
    
    print('')
    print('----------------------------------------------------')
    print(f" Models")
    print('----------------------------------------------------')

    for key in list(manifest):
        if manifest[key]['type'] != 'model':
            manifest.pop(key)
            
    pprint.pprint(manifest)

    print('')