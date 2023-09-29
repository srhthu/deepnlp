from dataclasses import dataclass
import json
from accelerate import Accelerator
from tqdm import tqdm
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

    num_epoch: int = 3
    max_steps: Optional[int] = None
    logging_steps: int = 10
    eval_steps: Union[int, float] = -1
    early_stop: Optional[bool] = None
    early_stop_patience: Union[int, float] = -2

    metric_for_best: Optional[str] = None
    greater_is_better: Optional[bool] = None

    # bellow are optional
    optim_name: Optional[str] = 'adamw'
    lr: float = 1e-5
    scheduler_type: Optional[str] = None

    @property
    def train_batch_size(self) -> int:
        return self.device_batch_size * self.n_gpu
    
    @property
    def n_gpu(self):
        # do not use nn.DataParallel
        return 1


class AccTrainer:
    """
    Accelerate based trainer.
    """
    def __init__(
        self,
        config:
    ):
        ...
    
    def train(self):
        ...