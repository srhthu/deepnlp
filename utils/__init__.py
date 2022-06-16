"""
Utilities for I/O, pytorch, machine learning, ...
"""

import os
import sys
import json
import time

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .tokenizer import (
    Tokenizer,
    Vocabulary,
    JiebaWordTokenize,
    TokenMergeLetterTokenize,
    CharacterTokenize
)

from .logger import Logger

TOKENIZER_MAPPING = {
    'jieba': JiebaWordTokenize,
    'token_m': TokenMergeLetterTokenize
}


# I/O
def read_json(path, **kwargs):
    # Read from a json file path.
    with open(path) as f:
        data = json.load(f, **kwargs)
    return data

def read_json_line(path, n_line = None):
    """
    Read a txt file with json lines.

    Input:
        n_line: number of lines to read. Useful for debug.
    """
    with open(path) as f:
        if n_line is None:
            data = [json.loads(k) for k in f]
        elif isinstance(n_line, int):
            data = [json.loads(f.readline()) for i in range(n_line)]
        else:
            raise ValueError(f'n_line should be int but get {n_line}')
    return data

def save_json(data, path, **kwargs):
    # Save Jsonified data to a path.
    with open(path, 'w') as f:
        json.dump(data, f, **kwargs)

def save_json_line(samples, path):
    """
    Save a list of samples to a path with Json format.
    """
    with open(path, 'w', encoding='utf8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii = False) + '\n')


# python dict
def zip_dict(**kwargs) -> List[Dict]:
    """
    A dict-version zip function.

    Input:
        kwargs whoes values are a List.
    """
    col_name = list(kwargs.keys())
    num_samples = len(kwargs[col_name[0]])
    
    sample_list = [{col: kwargs[col][i] for col in col_name} for i in range(num_samples)]
    return sample_list

def swap_dict_key(obj: dict, key_map: Dict[str, str]):
    """
    For a dict, change the key names in key_map with the new name.
    """
    new_obj = {}
    for k,v in obj.items():
        if k in key_map:
            k = key_map[k]
        new_obj[k] = v
    return new_obj

def slice_dict(obj:dict, keys: List[str]):
    """
    Retrieve a sub-dict with given keys
    """
    return {k:v for k,v in obj.items() if k in keys}


# torch
def to_cuda(data):
    """
    Move input tensors to cuda.

    Args:
        data (dict or list)
    """
    def _cuda(x):
        return x.cuda() if isinstance(x, torch.Tensor) else x

    if isinstance(data, dict):
        cuda_data = {k:_cuda(v) for k,v in data.items()}
        
    elif isinstance(data, list):
        cuda_data = [_cuda(k) for k in data]
    else:
        raise ValueError(f'data should be an instance of dict or list, but get {type(data)}')
    return cuda_data

def to_numerical(data:Dict[str, Union[torch.Tensor, float]]):
    def _numerical(x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            return x.item() if x.reshape(-1).shape[0] == 1 else x.numpy()
        else:
            return x
    num_data = {k:_numerical(v) for k,v in data.items()}
    return num_data

# numpy & torch
def concat(a, b):
    """
    Concatenate b to a on the first dim.

    Args:
        a: can be None. type: np.array or torch.Tensor
    """
    if a is None:
        return b
    else:
        if isinstance(a, np.ndarray):
            return np.concatenate([a,b], axis = 0)
        elif isinstance(a, torch.Tensor):
            return torch.cat([a,b], dim = 0)
        else:
            raise TypeError(f"Unsupported type for concatenation: got {type(a)}")

# time
def get_timestamp(year = False):
    """
    Return the time stamp. Useful for naming experiments.
    """
    fmt = ('%Y' if year else '') + '%m%d_%H%M'
    time_stamp = time.strftime(fmt, time.localtime())
    
    return time_stamp

# readibility
def obj_to_str(obj, **kwargs):
    if isinstance(obj, (int, float)):
        return numerical_to_str(obj)
    elif isinstance(obj, (tuple, list)):
        return '[' + ', '.join(map(obj_to_str, obj)) + ']'
    elif isinstance(obj, dict):
        sep = kwargs.get('sep', ' ')
        return sep.join([
            f'{k}: {obj_to_str(v, sep = sep)}' for k,v in obj.items()
        ])
    else:
        return str(obj)
        
def numerical_to_str(x: Union[int, float]):
    return f'{x:.5g}' if x > 0.001 and x < 1000 else f'{x:.3e}'


# make dir
def make_exp_dir(path_prefix, year = False):
    """
    Make a dir with timestamp. If conflicting, wait 30s and try again.
    """
    fail_count = 0
    while True:
        if fail_count >=3:
            raise OSError('Path exists after three attempts')
        
        out_path = path_prefix + '_' + get_timestamp(year)
        if os.path.exists(out_path):
            fail_count += 1
            print('Path conflict. Sleep 30s ...')
            time.sleep(30)
        else:
            os.makedirs(out_path, exist_ok = False)
            return out_path


# evaluation
def torch_acc_logits(logits: torch.Tensor, label: torch.Tensor
)->Tuple[torch.Tensor, torch.Tensor]:
    preds = torch.argmax(logits, dim = -1)
    acc = (preds == label).to(torch.float32).mean()

    return acc, preds