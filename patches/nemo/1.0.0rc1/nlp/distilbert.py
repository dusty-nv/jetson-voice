# Copyright 2020 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import DistilBertModel
from typing import Dict, Optional

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.core.classes import typecheck

__all__ = ['DistilBertEncoder']


class DistilBertEncoder(DistilBertModel, BertModule):
    """
    Wraps around the Huggingface transformers implementation repository for easy use within NeMo.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        These are ordered incorrectly in bert_module.py WRT to QAModel.forward()
        DistilBert doesn't use token_type_ids, but the QAModel still needs them during export.
        By re-ordring them, the correct input_names are used during export of the ONNX model.
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True)
        }

    '''
    # note:  disabling the token_type_ids here still leads to incorrect names, because QAModel.forward()
    #        still needs the token_type_ids to run the trace, and hence the input_example is still larger
    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return ['token_type_ids']
    '''
    
    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # distilBert does not use token_type_ids as the most of the other Bert models
        res = super().forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        return res
        