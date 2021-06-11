#!/usr/bin/env python3
# coding: utf-8

import sys
import pprint

from jetson_voice import (
    ASR, NLP, TTS, 
    AudioInput, AudioOutput, list_audio_devices,
    ConfigArgParser
)
       
parser = ConfigArgParser()

parser.add_argument('--asr-model', default='quartznet', type=str, help='ASR model')
parser.add_argument('--nlp-model', default='distilbert_intent', type=str, help='NLP model')
parser.add_argument('--tts-model', default='fastpitch_hifigan', type=str, help='TTS model')
parser.add_argument('--wav', default=None, type=str, help='path to input wav/ogg/flac file')
parser.add_argument('--mic', default=None, type=str, help='device name or number of input microphone')
parser.add_argument('--output-device', default=None, type=str, help='device name or number of audio output')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)
    
# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()
    
# load the models
tts = TTS(args.tts_model)
asr = ASR(args.asr_model, add_punctuation=False)
nlp = NLP(args.nlp_model)

if asr.classification:
    raise ValueError(f"'{args.asr_model}' is a classification model - must use a transcription model for agent")

if nlp.config.type != 'intent_slot':
    raise ValueError(f"'{args.nlp_model}' has type '{nlp.config.type}' - the agent requires an intent_slot model")
    
# create the audio streams
audio_input = AudioInput(wav=args.wav, mic=args.mic, 
                         sample_rate=asr.sample_rate, 
                         chunk_size=asr.chunk_size)

audio_output = AudioOutput(device=args.output_device,
                           sample_rate=tts.sample_rate)


def get_slot(results, name, default='', threshold=0, merge=True):
    """
    Retrieve a slot by name from the intent/slot results.
    The name can be a list of names, and any of them will be matched.
    Only slots with a score above the threshold will be returned.
    If merge is true, all slots by that name will be combined.
    If merge is false, the first matching slot will be returned.
    """
    if isinstance(name, str):
        name = [name]
        
    slots = []

    for slot in results['slots']:
        if any(slot['slot'] == n for n in name) and slot['score'] >= threshold:
            slots.append(slot['text'])
            
    if len(slots) == 0:
        return default
        
    if len(slots) > 1 and merge:
        return ' '.join(slots)
        
    return slots[0]
      
      
def generate_response(query):
    results = nlp(query)
    pprint.pprint(results)
    
    intent = results['intent']
    
    if intent == 'general_praise':
        return "Why thank you very much!"
        
    elif intent == 'weather_query':
        place = get_slot(results, 'place_name')
        date = get_slot(results, 'date')
        
        response = "The weather "
        
        if place: response += 'in ' + place + ' '
        if date:  response += date + ' '
        
        return response + "is forecast to be sunny with a high of 78 degrees."
        
    elif intent == 'recommendation_locations':
        place = get_slot(results, ['place_name', 'business_name'])
        
        if not place:
            return "Please ask again with the name of a store or restaurant."
          
        return f"{place} is located 1 mile away at 1 2 3 Main Street."
        
    return "I'm sorry, I don't understand."
    
# run agent
for input_samples in audio_input:
    transcripts = asr(input_samples)

    for transcript in transcripts:
        print(transcript['text'])
        
        if not transcript['end']:
            continue
            
        print('')
        
        response = generate_response(transcript['text'])
        print(response)
        
        audio_output.write(tts(response))

    """
    if transcripts[0] != 'unknown' and transcripts[1] != 'silence':
        response = generate_response(transcripts[0])
        print(response)
        
        audio_output.write(tts(response))
    """