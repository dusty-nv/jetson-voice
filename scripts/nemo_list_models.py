#!/usr/bin/env python3
# coding: utf-8

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts

print('nemo version:', nemo.__version__)

asr_archs = [model for model in dir(nemo_asr.models) if model.endswith("Model")]
nlp_archs = [model for model in dir(nemo_nlp.models) if model.endswith("Model")]
tts_archs = [model for model in dir(nemo_tts.models) if model.endswith("Model")]

print('ASR architectures:', asr_archs)  
print('NLP architectures:', nlp_archs)
print('TTS architectures:', tts_archs)

for asr_arch in asr_archs:
    print('')
    print('#####################################################')
    print('## nemo_asr.models.{:s}'.format(asr_arch))
    print('#####################################################')
    print(getattr(nemo_asr.models, asr_arch).list_available_models())

for nlp_arch in nlp_archs:
    print('')
    print('#####################################################')
    print('## nemo_nlp.models.{:s}'.format(nlp_arch))
    print('#####################################################')
    print(getattr(nemo_nlp.models, nlp_arch).list_available_models())
    
print('')
print('#####################################################')
print('## nemo_nlp.models.pretrained_lm_models')
print('#####################################################')
for model in nemo_nlp.modules.get_pretrained_lm_models_list():
    print(model)

for tts_arch in tts_archs:
    print('')
    print('#####################################################')
    print('## nemo_tts.models.{:s}'.format(tts_arch))
    print('#####################################################')
    print(getattr(nemo_tts.models, tts_arch).list_available_models())