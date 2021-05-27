#!/usr/bin/env python3
# coding: utf-8

import os
import json
import pprint
import logging
import argparse


#
# Default global configuration
#
# This can be overriden at runtime with command-line options (see ConfigArgParser)
# such as --global-config to load your own configuration from json file,
# or by calling config.load('my_config.json')
#
# You can also set the options directly on the 'config' object, e.g.
#
#    config.model_dir = '/path/to/my/models'
#    config.log_level = 'warning'
#
# It's recommended to use one of the methods above instead of changing _default_config directly.
#
_default_global_config = {
    'version' : 0.1,
    'model_dir' : 'data/networks',
    'model_manifest' : 'data/networks/manifest.json',
    'default_backend' : 'tensorrt',
    'log_level' : 'info',
    'debug' : False,
    'profile' : False
}


class ConfigDict(dict):
    """
    Configuration dict that can be loaded from JSON and has members
    accessible via attributes and can watch for updates to keys.
    """
    def __init__(self, *args, path=None, watch=None, **kwargs):
        """
        Parameters:
          path (str) -- Path to JSON file to load from
          
          watch (function or dict) -- A callback function that gets called when a key is set.
                                      Should a function signature like my_watch(key, value)
                                      This can also be a dict of key names and functions,
                                      and each function will only be called when it's particular
                                      key has been set.  You can also subclass ConfigDict and
                                      override the __watch__() member function.
        """                                
                                         
        super(ConfigDict, self).__init__(*args, **kwargs)
        
        self.__dict__['path'] = path
        self.__dict__['watch'] = watch
        
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.__watch__(y, x[y])
                    
        for x in kwargs:
            self.__watch__(x, kwargs[x])
               
        if path:
            self.load(path)
            
    def load(self, path, clear=False):
        """
        Load from JSON file.
        """
        from .resource import find_resource  # import here to avoid circular dependency
        
        path = find_resource(path)
        self.__dict__['path'] = path
        
        if clear:
            self.clear()
            
        with open(path) as file:
            config_dict = json.load(file)
        
        self.update(config_dict)
        
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return self[attr]
        
    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
        else:
            self[attr] = value
        
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigDict(value, watch=self.watch)
            
        super(ConfigDict, self).__setitem__(key, value)
        self.__watch__(key, value)
    
    def __watch__(self, key, value):
        #print(f'watch {key} -> {value}')

        if not self.watch:
            return
            
        if isinstance(self.watch, dict):
            if key in self.watch:
                self.watch[key](key, value)
        else:
            self.watch(key, value)
            
    def __str__(self):
        return pprint.pformat(self)
        
    #def __repr__(self):
    #    return pprint.saferepr(self)
        
    def setdefault(self, key, default=None):
        if isinstance(default, dict):
            value = ConfigDict(value, watch=self.watch)
        changed = key not in self
        value = super(ConfigDict, self).setdefault(key, default)
        if changed: self.__watch__(key, value)
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
        

#
# logging handlers
#
logging.basicConfig(format='[%(asctime)s] %(filename)s:%(lineno)d - %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO) 

global_config = None

def _set_log_level(key, value):
    log_value = value.upper()
    
    if log_value == 'VERBOSE':
        log_value = 'DEBUG'
        
    log_level = getattr(logging, log_value, None)
    
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {value}')
       
    logging.getLogger().setLevel(log_level)
    logging.debug(f'set logging level to {value}')

    if global_config is not None and value.upper() == 'DEBUG':
        global_config['debug'] = True
    
#
# global config definition
#
global_config = ConfigDict(_default_global_config, watch={'log_level':_set_log_level})

if global_config.log_level.upper() == 'DEBUG':
    global_config['debug'] = True
    
logging.debug(f'global config:\n{global_config}')


#
# custom arg parser
#
class ConfigArgParser(argparse.ArgumentParser):
    """
    ArgumentParser that provides global configuration options.
    """
    def __init__(self, *args, **kwargs):
        super(ConfigArgParser, self).__init__(*args, **kwargs)
    
        self.add_argument('--model-dir', default=_default_global_config['model_dir'], help=f"sets the root path of the models (default '{_default_global_config['model_dir']}')")
        self.add_argument('--model-manifest', default=_default_global_config['model_manifest'], help=f"sets the path to the model manifest file (default '{_default_global_config['model_manifest']}')")
        self.add_argument('--global-config', default=None, type=str, help='path to JSON file to load global configuration from')
        self.add_argument('--list-models', action='store_true', help='lists the available models (from $model_dir/manifest.json)')
        self.add_argument('--profile', action='store_true', help='enables model performance profiling')
        self.add_argument('--verbose', action='store_true', help='sets the logging level to verbose')
        self.add_argument('--debug', action='store_true', help='sets the logging level to debug')
        
        log_levels = ['debug', 'verbose', 'info', 'warning', 'error', 'critical']
        
        self.add_argument('--log-level', default=_default_global_config['log_level'], type=str, choices=log_levels,
                          help=f"sets the logging level to one of the options above (default={_default_global_config['log_level']})")
        
    def parse_args(self, *args, **kwargs):
        args = super(ConfigArgParser, self).parse_args(*args, **kwargs)
        
        global_config.model_dir = args.model_dir
        global_config.model_manifest = args.model_manifest
        global_config.log_level = args.log_level
        
        if args.profile:
            global_config.profile = True
            
        if args.verbose:
            global_config.log_level = 'verbose'
            
        if args.debug:
            global_config.log_level = 'debug'
        
        if args.global_config:
            global_config.load(args.global_config)
            
        if args.list_models:
            from .resource import list_models
            list_models()
            
        logging.debug(f'global config:\n{global_config}')    
        return args

