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
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator
from transformers.optimization import (
    AdamW,
    get_scheduler
)

from . import utils
from .trainer import trainer_utils

OPTIMIZER: Dict[str, torch.optim.Optimizer] = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

def simple_data_collator(features: List) -> Dict[str, Any]:
    first = features[0]
    batch = {}
    for k,v in first.items():
        if not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch

def second_to_str(sec):
    """
    convert seconds to hour:minute:second
    """
    sec = int(sec)
    m,s = divmod(sec, 60)
    h,m = divmod(m, 60)
    format_str = f'{m:02}:{s:02}'
    if h > 0:
        format_str = f'{h:02}:' + format_str
    return format_str

@dataclass
class BatchSize:
    """
    Calculate different types of batch size.

    Recommend set batch_size or batch_size + device_batch_size.

    Args:
        batch_size: logic optimization batch size
        device_batch_size: (maximum) batch_size on one device
        n_gpu: number of gpu.
        accmulate_step: Optional[int] = None
    
    Properties
        iter_batch_size: n_gpu * device_batch_size
    
    batch_size = n_device * device_batch_size * accumulate_step
    """
    batch_size: int
    device_batch_size: Optional[int] = None
    accumulate_step: Optional[int] = None
    n_gpu: Optional[int] = None
    use_cuda: Optional[bool] = None

    def __post_init__(self):
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available() # use if have
        if self.n_gpu is None and self.use_cuda:
            self.n_gpu = torch.cuda.device_count() # use all gpus
        n_device = self.n_gpu if self.use_cuda else 1
        
        bs_dev, mod = divmod(self.batch_size, n_device)
        if mod !=0:
            raise ValueError((
                f'batch_size{self.batch_size} cannot be devided exactly by'
                f'the number of device number{n_device}.')
            )
        
        # 2*2 cases
        if self.device_batch_size is not None and self.accumulate_step is not None:
            bs = n_device * self.device_batch_size * self.accumulate_step
            if bs != self.batch_size:
                raise ValueError(f'batch_size does not match other settings: {bs}')
        elif self.device_batch_size is not None:
            # infer accumulate step
            if bs_dev <= self.device_batch_size:
                self.device_batch_size = bs_dev
                self.accumulate_step = 1
            else:
                self.accumulate_step, mod = divmod(bs_dev, self.device_batch_size)
                if mod != 0:
                    raise ValueError(f'batch_size{self.batch_size} cannot be devided exactly.')
        elif self.accumulate_step is not None:
            self.device_batch_size, mod = divmod(bs_dev, self.accumulate_step)
            if mod != 0:
                raise ValueError(f'batch_size{self.batch_size} cannot be devided   exactly.')
        else:
            self.device_batch_size = bs_dev
            self.accumulate_step = 1 # do not accumulate gradient
    
    @property
    def n_device(self):
        return self.n_gpu if self.use_cuda else 1
    
    @property
    def iter_batch_size(self):
        """batch size for one iteration"""
        return self.batch_size // self.accumulate_step


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
    batch_size: Union[int, BatchSize],
    dataset,
    optimizer: torch.optim.Optimizer,
    device_batch_size: Optional[int] = None,
    accumulate_step: Optional[int] = None,
    collate_fn = simple_data_collator,
    max_grad_norm: Optional[float]= 3.0,
    compute_loss = default_feed,
    lr_scheduler = None,
    global_step = 0,
    logger = None,
    logging_steps: int =10,
    step_ave = None,
    show_bar = False,
    tb_writer: Optional[SummaryWriter] = None
):
    """
    For training one epoch, the lr_scheduler can be none.
    For training multiple epochs, the optimizers are passed from outside loops.
    """
    if not isinstance(batch_size, BatchSize):
        batch_size = BatchSize(batch_size, device_batch_size=device_batch_size,
            accumulate_step = accumulate_step)
        
    if collate_fn is None:
        collate_fn = simple_data_collator

    dl = DataLoader(dataset, batch_size = batch_size.iter_batch_size, shuffle = True,
        drop_last=True, collate_fn = simple_data_collator)

    if batch_size.use_cuda:
        model.cuda()

    if batch_size.n_device > 1:
        net = nn.DataParallel(model)
    else:
        net = model
    net.train()

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.disabled = True # do not log 

    # some constant
    acc_step = batch_size.accumulate_step
    train_params = [p for group in optimizer.param_groups for p in group['params']]

    # some variables
    timestamp = time.time()
    start = timestamp
    pct = 0
    if step_ave is None:
        step_ave = trainer_utils.DictNumberAverager()
    epoch_ave = trainer_utils.DictNumberAverager()

    net.zero_grad()
    for i, batch in tqdm(enumerate(dl), disable = not show_bar, total=len(dl)):
        if batch_size.use_cuda:
            batch = utils.to_cuda(batch)
        outputs = compute_loss(net, batch)

        if batch_size.n_device > 1:
            # get average for data parallel
            for k in outputs.keys():
                v = outputs[k]
                if isinstance(v, torch.Tensor) and v.numel() == batch_size.n_device:
                    outputs[k] = v.mean()
            #loss = loss.mean()
        loss = outputs['loss']
        
        loss = loss / acc_step
        loss.backward()

        

        metrics = utils.to_numerical(outputs)
        step_ave.record(metrics)
        epoch_ave.record(metrics)

        if (global_step+1) % acc_step == 0:
            # update gradient
            # clip gradient
            if (max_grad_norm and max_grad_norm > 0):
                nn.utils.clip_grad_norm_(train_params, max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            net.zero_grad()
            # log
            if ((global_step+1) // acc_step) % logging_steps == 0:
                logs = step_ave.average()
                if tb_writer is not None:
                    for name, value in logs.items():
                        tb_writer.add_scalar(f'{name}/train', value, (global_step+1) // acc_step)
                
                # minic tqdm bar
                duration = time.time() - timestamp
                timestamp = time.time()
                delta_p = (i+1) / len(dl) - pct
                pct = (i+1) / len(dl)
                past_time = second_to_str(time.time() - start)
                left_time = second_to_str(duration / delta_p * (1-pct))

                msg = ''.join((
                    f'Train {pct*100:>3.0f}% | ',
                    utils.obj_to_str(logs),
                    #f'|{i+1} [{past_time}<{left_time}, {duration:.2f}s/it]'
                    f' | {i+1} [{past_time}<{left_time}]'
                ))
                logger.debug(msg)
                
        global_step += 1
    
    # end epoch
    epoch_logs = epoch_ave.average()
    epoch_logs['time'] = f'{(time.time() - start) / 60:.2f}min'
    #logger.info('[End of Epoch]\n' + utils.obj_to_str(epoch_logs))
    return epoch_logs, global_step


def train_multiple_epochs(
    args: Union[dict, Namespace],
    model,
    train_dataset,
    dev_datasets: Optional[List[Dataset]] = None,
    compute_loss = None,
    eval_feed_fn = None,
    eval_metric_fn = None,
    logger = None,
    show_bar = True,
    to_tensorboard = True,
    save_best = True
):
    """
    Support evaluate on multiple datasets

    Args:
        args: necessary attributes:
            - batch_size


    """
    if logger is None:
        logger = utils.Logger(False)
    
    if not isinstance(args, dict):
        args = vars(args)
    batch_size = args['batch_size']
    device_batch_size = args.get('device_batch_size')
    accumulate_step = args.get('accumulate_step')
    logging_steps = args.get('logging_steps', 10)
    optim_name = args['optim_name']
    lr = args['lr']
    num_epoch = args['num_epoch']
    eval_batch_size = args.get('eval_batch_size', batch_size)
    output_dir = args.get('output_dir')
    early_stop = args.get('early_stop', False)
    early_stop_metric = args.get('early_stop_metric', 'acc')
    early_stop_patience = args.get('early_stop_patience', 2)

    if args['output_dir'] is not None and to_tensorboard:
        writer = SummaryWriter(os.path.join(args['output_dir'], 'tb'))
    else:
        writer = None

    bs = BatchSize(batch_size, device_batch_size = device_batch_size, 
        accumulate_step = accumulate_step)
    logger.info(repr(bs))

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
    epoch_eval_results = []
    for epi in range(num_epoch):
        # train
        logger.info(f'\nEpoch {epi + 1} Start')
        logs, global_step = train_one_epoch(
            model, bs, train_dataset, optimizer,
            compute_loss = compute_loss,
            lr_scheduler = lr_scheduler,
            global_step = global_step, 
            logger = logger,
            logging_steps = logging_steps,
            tb_writer = writer)
        logger.info(f'\nEpoch {epi + 1} End: {utils.obj_to_str(logs)}')

        if dev_datasets is None:
            continue
        if not isinstance(dev_datasets, list):
            dev_datasets = [dev_datasets,]
        # evaluate
        eval_metric_list = []
        msgs = []
        for i, dev_dataset in enumerate(dev_datasets):
            ev_name = 'valid' if i == 0 else 'test'
            outputs = do_predict(model, BatchSize(eval_batch_size),
                    dev_dataset, eval_feed_fn, show_bar = show_bar)
            metrics = eval_metric_fn(outputs)

            eval_metric_list.append(metrics)
            msg =  f'[Eval {ev_name} @{epi + 1}] {utils.obj_to_str(metrics)}'
            logger.info('\n' + msg)
            msgs.append(msg)
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f'{name}/{ev_name}', value, epi + 1)
        epoch_eval_results.append(eval_metric_list)
        es_metric = eval_metric_list[0][early_stop_metric]
        if es_metric > best_metric:
            logger.info(f'New best metric {epi + 1}')
            if writer is not None:
                writer.add_text('best metrics', ' '.join(msgs), epi + 1)
            best_metric = es_metric
            if early_stop:
                patience = 0
            # save model
            if output_dir is not None and save_best:
                torch.save(model.state_dict(), os.path.join(
                    args['output_dir'], 'best_model.bin'))
                # overwrite old one
        else:
            if early_stop:
                patience += 1
                if patience >= early_stop_patience:
                    logger.info(f'No improvement after {patience} epochs. Early stop')
                    break
    return epoch_eval_results


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
    args: BatchSize,
    dataset,
    feed_fn: Callable = default_feed_fn,
    show_bar: bool = True,
    collate_fn = simple_data_collator
    ):
    """
    the batch_size is device_batch_size.

    Args:
        feed_fn:
            args: (model, batch_data)
            return: List[Tensor] or Dict[str, Tensor]
    """
    assert args.accumulate_step == 1

    if collate_fn is None:
        collate_fn = simple_data_collator
    
    dl = DataLoader(dataset, batch_size = args.iter_batch_size, shuffle = False, collate_fn = collate_fn)

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
                        if outputs_np[i].ndim == 0:
                            continue
                        all_outputs[i] = np.concatenate((all_outputs[i], outputs_np[i]), axis = 0)
            elif isinstance(outputs, dict):
                outputs_np = {k: v.cpu().numpy() for k,v in outputs.items()}
                if all_outputs is None:
                    all_outputs = outputs_np
                else:
                    for k,v in outputs_np.items():
                        if v.ndim == 0:
                            continue
                        all_outputs[k] = np.concatenate((all_outputs[k], v), axis = 0)

    return all_outputs