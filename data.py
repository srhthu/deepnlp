"""
A general torch.Dataset supporting inference of data format.

Features:
    Support small load
"""
import os
import sys
import json
from collections import OrderedDict

import pickle
from typing import List

from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils import read_json_line, zip_dict


# =========== Dataset Classes ===========
class UniversalDataset(Dataset):
    """
    A universal dataset class.

    Support tuple and dict like instance.

    Supported initialization methods:
        - by sample. List of samples (`dict` or `list/tuple`)
        - by column. pass kwargs of columns or list of columns
    
    Attributes:
        dataset: List of samples.
    """
    def __init__(self, *args, **kwargs):
        self.dataset = None
        self._sample_type = None # 0: list; 1: dict
        self.key_mapping = None
        
        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError('No arguments found.')
        elif len(args) == 1 and len(kwargs) == 0:
            self._sample_type = self._get_sample_type(args[0][0])
            self.dataset = args[0]
        elif len(args) > 1 and len(kwargs) == 0:
            self._sample_type = 0
            self.dataset = [tuple(line) for line in zip(*args)]
        elif len(args) == 0:
            self._sample_type = 1
            
            self.dataset = zip_dict(**kwargs)
        else:
            raise ValueError('Arguments should be either positional or key words.')

    def decorate(self, sample):
        """
        Customize in the subclass or just assign the value.
        """
        if self.key_mapping is None:
            return sample
        else:
            mapped_sample = {}
            for k, map_k in self.key_mapping.items():
                mapped_sample[map_k] = sample[k]
            return mapped_sample
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.decorate(self.dataset[idx])
    
    def _get_sample_type(self, sample):
        """
        Sample type. 0: list, 1: dict.
        """        
        if isinstance(sample, list) or isinstance(sample, tuple):
            return 0
        elif isinstance(sample, dict):
            return 1
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}. Expected list, tuple and dict")