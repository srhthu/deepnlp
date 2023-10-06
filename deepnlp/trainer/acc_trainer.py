from dataclasses import dataclass
import json
from accelerate import Accelerator
from tqdm import tqdm
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.optimization import (
    AdamW,
    get_scheduler
)

@dataclass
class TrainingArgs:
    """
    General arguments for a trainer
    """
    batch_size: int = 16  # total batch size
    device_batch_size: int = None
    eval_batch_size: Optional[int] = None
    # the gradient_accumulate_step will be inferred

    deepspeed: Optional[Union[str, dict]] = None

    # flow control
    num_epoch: int = 3
    max_steps: Optional[int] = None
    logging_steps: int = 10 
    eval_steps: Optional[int]= None
    save_steps: Optional[int] = None
    eval_epochs: Optional[int] = None
    save_epochs: Optional[int] = None

    # early stop parameters
    early_stop: Optional[bool] = None
    early_stop_patience: Optional[int] = None
    metric_for_best: Optional[str] = None
    greater_is_better: Optional[bool] = None

    # bellow are optional
    optim_name: Optional[str] = 'adamw'
    lr: float = 1e-5
    scheduler_type: Optional[str] = None
    
    def __post_init__(self):
        # config deepspeed
        self.deepspeed_plugin = None
        if self.deepspeed:
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

            # Accelerate DeepSpeed Plugin
            from accelerate.utils import DeepSpeedPlugin

            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)


## Some utilities for training control
class EarlyStopController:
    """
    Record evaluation metrics, best step and determin early stop.
    """
    def __init__(
        self, 
        early_stop: bool, 
        patience: Optional[int] = None,
        metric_for_best: Optional[str] = None,
        greater_is_better: Optional[bool] = None
    ):
        if early_stop:
            assert patience and metric_for_best and (greater_is_better is not None)
        self.early_stop = early_stop
        self.patience = patience
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better

        self._step_metrics = []
        self._global_steps = []
        self._best_idx = -1
    
    def step(self, step, metrics):
        """
        Return whether current step is the new best
        """
        self._global_steps.append(step)
        self._step_metrics.append(metrics)

        if not self.early_stop:
            return None
        
        # determin whether is_best
        comp_metric = metrics[self.metric_for_best]
        is_best = False
        if self._best_idx < 0:
            is_best = True # this is the first record
        else:
            last_best = self._step_metrics[self._best_idx][self.metric_for_best]
            is_best = self.greater_is_better == (comp_metric > last_best)

        if is_best:
            self._best_idx = len(self._step_metrics) - 1
        return is_best

def group_nodecay_parameters(model, weight_decay = 0.0, no_decay = ['bias', 'LayerNorm.weight']):
    """
    Return parameter groups of decay parameters and no_decay parameters
    """
    named_params = list(model.named_parameters())
    nd_param_names = set([n for n, _ in named_params if any(nd in n for nd in no_decay)])
    param_groups = [
        {
            'params': [p for n,p in named_params if n not in nd_param_names],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n,p in named_params if n in nd_param_names],
            'weight_decay': 0.0
        }
    ]
    return param_groups

def get_smart_optimizer(model, lr, weight_decay = 0.0, **kws):
    param_groups = group_nodecay_parameters(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr = lr, **kws)
    return optimizer


class AccTrainer:
    """
    Accelerate based trainer.
    """
    def __init__(
        self,
        config: TrainingArgs,
        model: torch.Module,
        train_dataset,
        dev_dataset = None,
        test_dataset = None,
        output_dir = None,
        use_wandb: bool = False,
        optimizer = None,
        scheduler = None
    ):
        self.config = config
        
        accelerator = Accelerator(deepspeed_plugin = self.config.deepspeed_plugin)
        if optimizer is None:
            self.optimizer = get_smart_optimizer(model, lr = self.config.lr)
    
    def train(self):
        ...