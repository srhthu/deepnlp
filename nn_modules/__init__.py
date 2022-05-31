import os
import numpy as np
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq2vec import *
from .seq_encoder import *
from .seq2seqvec import *
from .torch_base import *
from .const import *

#print('load module __init__')
MODEL_MAPPING = {
    'mlp': MLP,
    'embedding': nn.Embedding,
    'seq2seqvec_cnn': Seq2SeqVec_CNN,
    'seq2seqvec_rnn': Seq2SeqVec_RNN,
    'seqpool': SeqPool
}
# model name is lowercase.

def build_model(mod_name:str, mod_args: Union[dict, list]) -> nn.Module:
    mod_cls = get_model_by_name(mod_name)
    if isinstance(mod_args, list):
        return mod_cls(*mod_args)
    else:
        return mod_cls(**mod_args)

def get_model_by_name(mod_name:str) -> Type[nn.Module]:
    mod_name = mod_name.lower()
    if mod_name not in MODEL_MAPPING:
        raise ValueError(f'Unknown model type: {mod_name}')
    return MODEL_MAPPING[mod_name]

# A sequential model.
class StackedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modlist = nn.ModuleList()
        for lay_name, lay_args in layers:
            modlist.append(build_model(lay_name, lay_args))
        self.modlist = modlist
    
    def forward(self, input, *args, **kwargs):
        output = self.modlist[0](input, *args, **kwargs)
        for mod in self.modlist[1:]:
            if isinstance(output, list):
                output = mod(*output)
            elif isinstance(output, dict):
                output = mod(**output)
            else:
                output = mod(output)
        return output
    
    def __str__(self):
        return str(self.modlist)
    

