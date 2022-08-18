"""
Utilities for the Trainer class.
"""

from lib2to3.pgen2.token import OP
import logging
import random
from tokenize import Number
import numpy as np
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

def initialize_logger(name = __name__, log_path = None):
    logger = logging.getLogger(name)
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(logging.Formatter(fmt = ""))
    logger.addHandler(stream_hd)
    if log_path is not None:
        file_hd = logging.FileHandler(log_path, mode = 'a')
        file_hd.setFormatter(logging.Formatter(fmt = ""))
        logger.addHandler(file_hd)
    logger.setLevel(logging.DEBUG)
    return logger

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe even if cuda is not available

@dataclass
class TrainerState:
    """
    Trainer state variables
    """
    epoch: Optional[float] = None
    epoch_n: int = 0
    global_step: int = 0
    max_steps: int = 0
    best_metric = None
    best_step = 0
    early_stopping_patience_conter = 0

@dataclass
class TrainerControl:
    should_save: bool = False
    should_log: bool = False
    should_evaluate: bool = False
    should_training_stop: bool = False

class NumberAverager:
    """
    Maintain a list of numbers and calculate their average.
    """
    def __init__(self):
        self.values = []
    def append(self, x):
        self.values.append(x)
    def _reset(self):
        self.values = []
    def average(self, reset = True):
        ave = np.mean(self.values)
        if reset:
            self._reset()
        return ave

class DictNumberAverager:
    """
    A dict of cls NumberAverager
    """
    def __init__(self, names: Optional[list] = None):
        self.names = names
        self._create()
    
    def _create(self):
        if self.names:
            self.averagers = {name:NumberAverager for name in self.names}
        else:
            self.averagers = {}
    
    def record(self, msg: dict):
        for name, value in msg.items():
            if not isinstance(value, (int, float)):
                # skip
                continue
            if name not in self.averagers:
                self.averagers[name] = NumberAverager()
            self.averagers[name].append(value)
    def _reset(self):
        self._create()
    
    def average(self, reset = True):
        ave_dict = {name: k.average(reset) for name, k in self.averagers.items()}
        return ave_dict


def compute_accuracy_with_logits(all_logits, all_labels):
    """
    Metric should start with `eval_`
    """
    all_preds = np.argmax(all_logits, axis = -1)
    num_samples = len(all_preds)
    acc = (all_preds == all_labels).sum() / num_samples
    metrics = {
        'eval_acc': acc
    }
    return metrics

def compute_multi_label_f1(all_logits, all_labels):
    all_preds = (all_logits > 0).astype(np.int)
    hit = 0
    n_pred = 0
    n_ground = 0
    for pred, label in zip(all_preds, all_labels):
        n_pred += sum(pred)
        n_ground += sum(label)
        hit += sum([(a==1 and b==1) for a,b in zip(pred,label)])
    
    prec = hit / n_pred if n_pred > 0 else 0.0
    recall = hit / n_ground if n_ground > 0 else 0.0
    if prec == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = prec * recall / (prec + recall)

    return {
        'eval_f1': f1,
        'eval_p': prec,
        'eval_r': recall
    }

def torch_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch