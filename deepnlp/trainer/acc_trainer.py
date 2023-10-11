from dataclasses import dataclass
import json
import numpy as np
from collections import defaultdict
from accelerate import Accelerator, PartialState
from tqdm import tqdm
import time
import os
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import logging
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
from transformers.trainer_pt_utils import get_model_param_count, nested_concat, nested_detach, nested_numpify

LOGGER_NAME = 'mytrainer'

@dataclass
class TrainingArgs:
    """
    General arguments for a trainer
    """
    batch_size: int = 16  # total batch size
    device_batch_size: int = None
    eval_batch_size: Optional[int] = None # per device
    # the gradient_accumulate_step will be inferred
    max_grad_norm = 3.0

    deepspeed: Optional[Union[str, dict]] = None

    # flow control
    num_epoch: int = None
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
        state = PartialState()
        if self.eval_batch_size is None:
            self.eval_batch_size = self.device_batch_size
        # infer gradient accumulate step
        if self.device_batch_size * state.num_processes > self.batch_size:
            # decrease the actual device_batch_size
            self.device_batch_size, r = divmod(self.batch_size, state.num_processes)
            print(f'reset device_batch_size to {self.device_batch_size}')
            g_acc_step = 1
        else:
            g_acc_step, r = divmod(self.batch_size, self.device_batch_size * state.num_processes)
        assert r == 0, (
            f"Cannot solve gradient accumulation step. batch_size={self.batch_size},"
            f"device_batch_size={self.device_batch_size}, n_proc={state.num_processes}\n"
        )
        self.gradient_accumulation_steps = g_acc_step
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
    
    @property
    def should_stop(self):
        if not self.early_stop:
            return False
        no_improve_step = len(self._global_steps) - (self._best_idx + 1)
        return (no_improve_step >= self.patience)

class AverageTensors:
    """
    To hold step outputs, only keep scales and output their average.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.records = None
        self.step_count = 0

    def filter_tensor(self, x):
        """keep tensors with only one element"""
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return True
        return False

    def record(self, tensor_dt):
        # do not merge
        new_dt = {k:v.detach().cpu().squeeze() for k,v in tensor_dt.items() if self.filter_tensor(v)}
        if self.records is None:
            self.records = new_dt
        else:
            for k,v in new_dt.items():
                self.records[k] += v
        self.step_count += 1
    
    def average(self)->Dict[str, float]:
        """Return average and reset history records"""
        ave_ts = {k:(v / self.step_count).item() for k,v in self.records.items()}
        self.reset()
        return ave_ts

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

def number2str(x: Union[int, float]):
    if abs(x) > 0.001 and abs(x) < 1000:
        return f'{x:.5g}'
    elif abs(x) < 1e-6:
        return '0.0'
    else:
        return f'{x:.3e}'

class DistLogger:
    """Only log on the local main process"""
    def __init__(self, name, output_dir):
        self.logger = logging.getLogger(name)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(10)
        self.output_dir = output_dir

        self.state = PartialState()
    
    def info(self, msg):
        if self.state.is_local_main_process:
            self.logger.info(msg)



## Trainer
class AccTrainer:
    """
    Accelerate based trainer.
    """
    def __init__(
        self,
        config: TrainingArgs,
        model: torch.nn.Module,
        train_dataset,
        eval_dataset = None,
        test_dataset = None,
        collate_fn = None,
        output_dir = None,
        use_wandb: bool = False,
        optimizer = None,
        scheduler = None,
        compute_metrics: Optional[Callable[[Dict[str, np.ndarray],Dict[str, np.ndarray]], Dict[str, float]]] = None
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer if optimizer \
            else get_smart_optimizer(model, lr = self.config.lr)
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn if collate_fn else default_data_collator
        self.compute_metrics = compute_metrics
        
        self.init_accelerator()

        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.logger = DistLogger('mytrainer', output_dir)
    
    def get_train_dataloader(self):
        dl = DataLoader(
            self.train_dataset, 
            batch_size = self.config.device_batch_size,
            shuffle=True,
            drop_last = True,
            collate_fn = self.collate_fn
        )
        return dl

    def init_accelerator(self):
        self.accelerator = Accelerator(
            deepspeed_plugin = self.config.deepspeed_plugin,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

    def train(self):
        # prepare
        config = self.config
        g_acc_step = config.gradient_accumulation_steps
        logger = self.logger
        
        train_dl = self.get_train_dataloader()
        model, train_dl, optimizer, scheduler = self.accelerator.prepare(
            self.model, train_dl, self.optimizer, self.scheduler
        )
        self.log(f'Wrap model type: {type(model)}')

        # determin max steps
        iter_per_epoch = len(train_dl) / g_acc_step
        num_epoch = config.num_epoch
        if num_epoch is None:
            max_steps = config.max_steps
            num_epoch = math.ceil(max_steps / iter_per_epoch)
        else:
            max_steps = iter_per_epoch * num_epoch # num of optimization step
        
        if any([config.eval_steps, config.eval_epochs]):
            eval_steps = config.eval_steps or iter_per_epoch * config.eval_epochs
        else:
            eval_steps = None
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {max_steps * config.batch_size:,}")
        logger.info(f"  Steps per epoch: {iter_per_epoch}")
        logger.info(f"  Num Epochs = {num_epoch:,}")
        logger.info(f"  Batch size per device = {config.device_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {g_acc_step}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
        
        # variables
        self.global_step = 0
        self.es_helper = EarlyStopController(
            config.early_stop, 
            config.early_stop_patience, 
            config.metric_for_best, 
            config.greater_is_better
        )
        total_batched_samples = 0
        tr_metrics = AverageTensors()
        # start training
        model.train()
        training_bar = tqdm(
            total = max_steps, dynamic_ncols= True, 
            disable = not self.accelerator.is_local_main_process
        )
        for epoch in range(num_epoch):
            # epoch begin
            for step, batch in enumerate(train_dl):
                total_batched_samples += 1
                # forward and backward
                with self.accelerator.accumulate(model):
                    outputs = self.training_step(model, batch)
                tr_metrics.record(outputs)
        
                if total_batched_samples % g_acc_step == 0:
                    # optimization step
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), 
                        config.max_grad_norm
                    )
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    model.zero_grad()
                    # step end
                    self.global_step += 1
                    training_bar.update(1)
                    
                    if self.global_step % config.logging_steps == 0:
                        # logging
                        tr_logs = tr_metrics.average()
                        tr_logs['step'] = self.global_step
                        tr_logs['epoch'] = epoch + step / len(train_dl)
                        self.log(tr_logs, tqdm_bar = training_bar)
                    
                    if eval_steps and (self.global_step % eval_steps == 0):
                        # evaluate
                        self.do_evaluate()
                    
                    if self.es_helper.should_stop or self.global_step >= max_steps:
                        break

            # epoch end
            if self.es_helper.should_stop or self.global_step >= max_steps:
                break
        # training end
        training_bar.close()

    def training_step(self, model, batch)->Dict[str, torch.Tensor]:
        """
        Prepare inputs, forward and backward.
        """
        outputs = model(**batch)
        loss = outputs['loss']
        self.accelerator.backward(loss)
        return outputs

    def compute_loss(self, model, batch)-> Dict[str, torch.Tensor]:
        """
        Return a dict that mush contain loss
        """
        return model(**batch)

    def _log(self, msg: str, tqdm_bar = None):
        if not self.accelerator.is_local_main_process:
            return None
        if tqdm_bar is not None:
            tqdm_bar.write(msg)
        else:
            self.logger.info(msg)
    def log(self, logs: Union[str, Dict[str, float]], tqdm_bar = None):
        if isinstance(logs, dict):
            logs = {k: number2str(v) for k,v in logs.items()}
        self._log(str(logs), tqdm_bar)

        # TODO: wandb
    
    def do_evaluate(self):
        if not self.eval_dataset:
            return None
        
        # prepare
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        if self.accelerator is None:
            self.accelerator = Accelerator()
        model, eval_dl = self.accelerator.prepare(self.model, eval_dataloader)
        all_preds_host = None
        all_inputs_host = None
        for batch in eval_dl:
            with torch.no_grad():
                preds = self.compute_preds(model, batch)

            all_preds, all_inputs = self.accelerator.gather_for_metrics((preds, batch))
            all_preds_host = (all_preds if all_preds_host is None 
                              else nested_concat(all_preds_host, all_preds))
            all_inputs_host = (all_inputs if all_inputs_host is None 
                               else nested_concat(all_inputs_host, all_inputs))
        
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                nested_numpify(all_preds_host), nested_numpify(all_inputs_host)
            )
        else:
            metrics = {'loss': all_preds_host['loss'].mean().tolist()}

        self.log(metrics)

        if hasattr(self, 'es_helper'):
            self.es_helper.step(self.global_step, metrics)
    
    
    def compute_preds(self, model, batch):
        """Evaluation feed forward function"""
        return model(**batch)

    def get_eval_dataloader(self, dataset):
        dl = DataLoader(
            dataset, 
            batch_size = self.config.device_batch_size,
            collate_fn = self.collate_fn
        )
        return dl

        