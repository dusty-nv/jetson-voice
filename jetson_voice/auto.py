#!/usr/bin/env python3
# coding: utf-8

from jetson_voice.asr import ASR
from jetson_voice.nlp import IntentSlot, QuestionAnswer, TextClassification, TokenClassification
from jetson_voice.tts import TTS

from jetson_voice.utils import load_resource


def AutoModel(resource, domain=None, *args, **kwargs):
    """
    Factory for automatically loading models and services.
    First the config is loaded and the type is checked.
    Then the correct instance for the resource is created.
    
    If a domain string is supplied (e.g. 'asr', 'nlp', 'tts'),
    then only resources from that domain will be created.
    """
    type_map = {
        # models
        'asr' : (ASR, 'asr'),
        'asr_classification' : (ASR, 'asr'),
        'intent_slot' : (IntentSlot, 'nlp'),
        'qa' : (QuestionAnswer, 'nlp'),
        'text_classification' : (TextClassification, 'nlp'),
        'token_classification' : (TokenClassification, 'nlp'),
        'tts': (TTS, 'tts'),
        
        # services
        'jarvis_asr' : (ASR, 'asr')
    }

    config = load_resource(resource, None, *args, **kwargs)
    
    if 'type' not in config:
        raise ValueError(f"'type' setting missing from config '{config.path}'")
        
    if config.type not in type_map:
        raise ValueError(f"'{config.path}' has invalid 'type' ({config.type})")
    
    if domain:
        if type_map[config.type][1] != domain.lower():
            raise ValueError(f"invalid model selected - '{config.path}' has domain '{type_map[config.type][1]}', but AutoModel() was called with domain={domain}")
            
    return type_map[config.type][0](config, *args, **kwargs)
