"""
Training arguments.

Think:
    what should this include?
    Store all arguments together or separately?

        Flow control: output, log, eval save
        initialize dataloader // better feed in
        initialize optimizer and scheduler // better feed in
        total step
        early stop?
        eval metric
        use gpu
        [] how to calculate loss?

"""

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class TrainingArguments:
    """
    Training Arguments.

    Parameters:
        output_dir (str, optional):
            If None, do not save and log
        do_eval (bool):
            Whether do evaluation during training
        evaluation_strategy (str):
            possible values:
                "no": No evaluation
                "steps": eval is done every `eval_steps`
                "epoch": eval is done at every epoch.
    """
    output_dir: Optional[str] = None
    do_eval: bool = False

    # log evaluate and save
    logging_steps: Union[int, float] = 100
    eval_steps: Optional[int] = None
    eval_epochs: Optional[int] = None
    save_steps: Optional[int] = None
    save_epochs: Optional[int] = None
    save_total_limit: Optional[int] = 3

    # training
    ## per device
    train_batch_size_per_device: int = 8
    eval_batch_size_per_device: int = 18
    n_gpu = None # if not set, initialize after __init__
    gradient_accumulation_steps = 1
    dataloader_num_workers: int = 0

    ## optimizer & scheduler
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0    

    ## total step
    num_train_epochs: float = 3.0
    max_steps: int = -1 # if > 0, prior to epochs
    
    ## eval metric
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = True

    ## early stoip
    early_stopping_patience: Optional[int] = 3 # none for no early stop

    no_cuda: bool = False
    seed: int = 26

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    @property
    def train_batch_size(self):
        dev_n = self.n_gpu if self.n_gpu > 0 else 1
        return dev_n * self.train_batch_size_per_device
    
    @ property
    def eval_batch_size(self):
        dev_n = self.n_gpu if self.n_gpu > 0 else 1
        return dev_n * self.eval_batch_size_per_device
    
    def __post_init__(self):
        if not self.eval_steps and not self.eval_epochs:
            self.eval_epochs = 1
        if not self.save_steps and not self.save_epochs:
            self.save_epochs = 1
        
        if self.n_gpu is None:
            self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
    
    def __str__(self):
        self_as_dict = asdict(self)

        attrs_as_str = [f"{k}={repr(v)},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


def get_epoch_trainer(**kwargs):
    args_kw = {
        'eval_epochs': 1,
        'save_epochs': 1
    }
    args_kw.update(kwargs)
    return TrainingArguments(**args_kw)