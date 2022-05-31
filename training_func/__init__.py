"""
training_func provides functiona utilities for training and evaluation loops.

Func:
    training_one_epoch: train one epoch.
        return (loss, other metrics...)
        no save, do log.
        support data parallel, gradient accumulation.
"""

import os
import sys
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import numpy as np
from tqdm import tqdm
import random
import time
import shutil
from pathlib import Path
import re

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

from .. import utils
from ..trainer import trainer_utils

OPTIMIZER: Dict[str, torch.optim.Optimizer] = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': AdamW,
}


@dataclass
class TrainingArgumentsForLoop:
    """
    batch_size:
        device_batch_size
        iter_batch_size
        optim_batch_size

    By default, can only set batch_size and infer other variables.
    Recommend set device_batch_size and infer accumulation_step if change between different devices.
    """
    batch_size: int
    n_gpu: Optional[int] = None
    device_batch_size: Optional[int] = None
    use_cuda: Optional[bool] = None
    gradient_accumulation_step: Optional[int] = None

    max_grad_norm: Optional[float] = 3.0
    logging_steps: int = 10  # step of optimization

    def __post_init__(self):
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available() # use if have
        if self.n_gpu is None and self.use_cuda:
            self.n_gpu = torch.cuda.device_count() # use all gpus
        n_device = self.n_gpu if self.use_cuda else 1
        mod = 0
        if self.device_batch_size is not None and self.gradient_accumulation_step is not None:
            bs = n_device * self.device_batch_size * self.gradient_accumulation_step
            if bs != self.batch_size:
                raise ValueError(f'batch_size does not match other settings: {bs}')
        elif self.device_batch_size is not None:
            self.gradient_accumulation_step, mod = divmod(
                self.batch_size, self.n_gpu * self.device_batch_size)
        elif self.gradient_accumulation_step is not None:
            self.device_batch_size, mod = divmod(
                self.batch_size, self.n_gpu * self.gradient_accumulation_step)
        else:
            self.device_batch_size, mod = divmod(self.batch_size, self.n_gpu)
            self.gradient_accumulation_step = 1 # do not accumulate gradient
        if mod != 0:
            raise ValueError(f'batch_size cannot be devided exactly: mod {mod}')
    
    @property
    def n_device(self):
        return self.n_gpu if self.use_cuda else 1
    
    @property
    def i_batch_size(self):
        """batch size for one iteration"""
        return self.batch_size // self.gradient_accumulation_step

def default_feed(model: nn.Module, batch_data: Dict[str, torch.Tensor]):
    """
    return
        Dict:
            loss: torch.Tensor
            [other_keys]: object to print
    """
    outputs =  model(**batch_data)
    return outputs

def train_one_epoch(
    model,
    training_args: TrainingArgumentsForLoop,
    dataset,
    compute_loss = default_feed,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler = None,
    global_step = 0,
    logger = None,
    show_bar = False
):
    """
    For training one epoch, the lr_scheduler can be none.
    For training multiple epochs, the optimizers are passed from outside loops.
    """

    dl = DataLoader(dataset, batch_size = training_args.i_batch_size, shuffle = True,
        drop_last=True, collate_fn = default_data_collator)

    if training_args.use_cuda:
        model.cuda()

    if training_args.n_device > 1:
        net = nn.DataParallel(model)
    else:
        net = model
    net.train()

    if logger is None:
        logger = utils.Logger(False) # do not log

    

    # some constant
    acc_step = training_args.gradient_accumulation_step
    max_grad_norm = training_args.max_grad_norm
    train_params = [p for group in optimizer.param_groups for p in group['params']]

    # some variables
    timestamp = time.time()
    start = timestamp
    step_ave = trainer_utils.DictNumberAverager()
    total_ave = trainer_utils.DictNumberAverager()

    net.zero_grad()
    for i, batch in tqdm(enumerate(dl), disable = not show_bar, total=len(dl)):
        if training_args.use_cuda:
            batch = utils.to_cuda(batch)
        outputs = compute_loss(net, batch)
        loss = outputs['loss']

        if training_args.n_device > 1:
            loss = loss.mean()
        
        loss = loss / acc_step
        loss.backward()

        metrics = utils.to_numerical(outputs)
        step_ave.record(metrics)
        total_ave.record(metrics)

        if (global_step+1) % acc_step == 0:
            # update gradient
            # clip gradient
            if (max_grad_norm and max_grad_norm > 0):
                nn.utils.clip_grad_norm_(train_params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            net.zero_grad()
            if ((global_step+1) // acc_step) % training_args.logging_steps == 0:
                logs = step_ave.average()
                duration = time.time() - timestamp
                logs['time'] = f'{duration:.2f}s'
                pct = (i+1) / len(dl) * 100
                logger.info(f'| {i+1:>} {pct:>3.0f}% | ' + utils.obj_to_str(logs))
                timestamp = time.time()
        global_step += 1
    
    # end epoch
    epoch_logs = total_ave.average()
    epoch_logs['time'] = f'{(time.time() - start) / 60:.2f}min'
    #logger.info('[End of Epoch]\n' + utils.obj_to_str(epoch_logs))
    return epoch_logs, global_step


def train_multiple_epochs(args, model, train_dataset, dev_dataset, compute_loss,
    eval_feed_fn, eval_metric_fn,
    logger = None, show_bar = True):
    if logger is None:
        logger = utils.Logger(False)
    
    batch_size = args['batch_size']
    logging_steps = args['logging_steps']
    optim_name = args['optim_name']
    lr = args['lr']
    num_epoch = args['num_epoch']
    eval_batch_size = args['eval_batch_size']
    output_dir = args['output_dir']
    early_stop = args['early_stop']
    early_stop_metric = args['early_stop_metric']
    early_stop_patience = args['early_stop_patience']

    train_args = TrainingArgumentsForLoop(batch_size = batch_size, logging_steps = logging_steps)
    optimizer = OPTIMIZER[optim_name.lower()](
        model.parameters(),
        lr = lr
    )
    steps_one_epoch = math.ceil(len(train_dataset) / batch_size)
    # [TODO] support other scheduler type.
    lr_scheduler = get_scheduler('linear', optimizer, 
        num_warmup_steps = 0, num_training_steps = steps_one_epoch * num_epoch)
    
    global_step = 0
    best_metric = 0
    patience = 0
    for epi in range(num_epoch):
        # train
        logger.info(f'[Epoch {epi + 1} Start]')
        logs, global_step = train_one_epoch(model, train_args, train_dataset, compute_loss, 
            optimizer, lr_scheduler, global_step = global_step, logger = logger)
        logger.info(f'[Epoch {epi + 1} End] {utils.obj_to_str(logs)}')
        # evaluate
        outputs = do_predict(model, TrainingArgumentsForLoop(eval_batch_size),
                dev_dataset, eval_feed_fn, show_bar = show_bar)
        metrics = eval_metric_fn(outputs)
        logger.info(f'[Eval {epi + 1}] {utils.obj_to_str(metrics)}')

        if early_stop:
            es_metric = metrics[early_stop_metric]
            if es_metric > best_metric:
                logger.info(f'New best metric {epi + 1}')
                best_metric = es_metric
                patience = 0
                # save model
                if output_dir is not None:
                    torch.save(model.state_dict(), os.path.join(
                        args['output_dir'], 'best_model.bin'))
                    # overwrite old one
            else:
                patience += 1
                if patience >= early_stop_patience:
                    logger.info(f'No improvement after {patience} epochs. Early stop')
                    break


def default_feed_fn(model, batch):
    outputs = model(**batch)

    # add label
    label = batch.get('label', None)
    if label is not None:
        if isinstance(outputs, (list, tuple)):
            outputs = (*outputs, label)
        elif isinstance(outputs, dict):
            outputs['label'] = label
    return outputs

def do_predict(
    model: nn.Module,
    args: TrainingArgumentsForLoop,
    dataset,
    feed_fn: Callable = default_feed_fn,
    show_bar: bool = True
    ):
    """
    the batch_size is device_batch_size.

    Args:
        feed_fn:
            args: (model, batch_data)
            return: List[Tensor] or Dict[str, Tensor]
    """
    assert args.gradient_accumulation_step == 1

    dl = DataLoader(dataset, batch_size = args.i_batch_size, shuffle = False, collate_fn = default_data_collator)

    if args.use_cuda:
        model.cuda()

    if args.n_device > 1:
        net = nn.DataParallel(model)
    else:
        net = model
    net.eval()

    all_outputs = None
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl), disable = not show_bar, total = len(dl)):
            if args.use_cuda:
                batch = utils.to_cuda(batch)
            
            outputs = feed_fn(net, batch)
            # outputs: List[Tensor] or Dict[str, Tensor]
            # Tensor has minimum dimensions of 1

            if isinstance(outputs, (list, tuple)):
                outputs_np = list(map(lambda k: k.cpu().numpy(), outputs))
                if all_outputs is None:
                    all_outputs = outputs_np
                else:
                    for i in range(len(all_outputs)):
                        all_outputs[i] = np.concatenate((all_outputs[i], outputs_np[i]), axis = 0)
            elif isinstance(outputs, dict):
                outputs_np = {k: v.cpu().numpy() for k,v in outputs.items()}
                if all_outputs is None:
                    all_outputs = outputs_np
                else:
                    for k,v in outputs_np.items():
                        all_outputs[k] = np.concatenate((all_outputs[k], v), axis = 0)

    return all_outputs