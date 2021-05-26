#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import importlib

from .config import global_config, ConfigDict


def find_resource(path):
    """
    Find a resource by checking some common paths.
    """
    if os.path.exists(path):
        return path
        
    search_dirs = [global_config.model_dir]
    
    for search_dir in search_dirs:
        search_path = os.path.join(search_dir, path)
        
        if os.path.exists(search_path):
            return search_path
    
    raise IOError(f"failed to locate resource '{path}'")


def load_resource(resource, factory_map, *args, **kwargs):
    """
    Load a resource instance.
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
            config = ConfigDict(backend=resource)
            
    elif isinstance(resource, dict):
        config = ConfigDict(resource)
    else:
        raise ValueError(f"expected string or dict type, instead got {type(resource).__name__}")
    
    config.setdefault('backend', global_config.default_backend)
    
    if config.backend not in factory_map:
        raise ValueError(f"'{config.path}' has invalid backend '{config.backend}' (valid options are: {', '.join(factory_map.keys())})")
        
    class_name = factory_map[config.backend].rsplit(".", 1)
    class_type = getattr(importlib.import_module(class_name[0]), class_name[1])
    
    logging.debug(f"creating instance of {factory_map[config.backend]} for '{config.path}' (backend {config.backend})")
    logging.debug(class_type)
    
    return class_type(config, *args, **kwargs)
    
    