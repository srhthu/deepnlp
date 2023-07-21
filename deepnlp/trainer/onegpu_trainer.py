import os
import sys
import json
import logging
from tokenize import Name
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
from dataclasses import dataclass

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

from .onegpu_utils import (
    infer_batch_sizes, to_cuda, outputs_to_number, 
    obj_to_str, DictNumberAverager, default_feed, 
    simple_data_collator, OPTIMIZER
)

@dataclass
class TrainingArgs:
    """
    General arguments for a trainer
    """
    batch_size: int = 16
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

class OneGPUTrainer:
    """
    Training models on one GPU. Eval by steps.

    Do not support DataParallel

    Attributes:
        - output_dir: if not None, should exist

    Training arguments:
        - eval_steps: iter number if positive else epoch number if negative
        - early_stop_patience: same as above. wait > patience then stop
    """
    def __init__(
        self,
        args,
        model,
        train_dataset,
        dev_dataset = None,
        test_dataset = None,
        output_dir = None,
        logger = None,
        writer: Optional[SummaryWriter] = None,
        optimizer = None,
        scheduler = None
    ):
        if isinstance(args, (Namespace, TrainingArgs)):
            args = vars(args)
        
        self.args = args.copy() # changes on self.args do not change original args
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.disabled = True # Do not log. Just a plaseholder
        self.logger = logger
        self.writer = writer

        self.global_step = 0
        self.best_log_step = 0

        # mandatory args
        self.batch_size = self.args['batch_size']
        self.device_bs, self.acc_step = self.infer_batch_size()
        self.eval_batch_size = self.args['eval_batch_size']
        if self.eval_batch_size is None:
            self.eval_batch_size = self.device_bs

        self.num_epoch = self.args['num_epoch']
        self.logging_steps = self.args['logging_steps']
        self.eval_steps = self.args['eval_steps']
        self.early_stop = self.args['early_stop']
        self.early_stop_patience = self.args['early_stop_patience']

        self.metric_for_best = self.args['metric_for_best']
        self.greater_is_better = self.args['greater_is_better']
        if self.greater_is_better is None:
            self.greater_is_better = not 'loss' in self.metric_for_best

        self.optimizer = optimizer if optimizer else self.get_optimizer()
        self.lr_scheduler = scheduler if scheduler else self.get_scheduler()

    def infer_batch_size(self):
        """
        batch_size = device_batch_size * 1 * acc_step
        """
        batch_size = self.args['batch_size']
        device_batch_size = self.args['device_batch_size']
        if (device_batch_size is None or
            device_batch_size  > batch_size):
            # do not need accumulation.
            device_batch_size = batch_size
        
        acc_step, mod = divmod(batch_size, device_batch_size)
    
        if mod != 0:
            raise ValueError((
                f'batch_size={batch_size} cannot be deviced '
                f'by device_batch_size={device_batch_size}')
            )
        
        assert batch_size == device_batch_size * acc_step

        return device_batch_size, acc_step
    
    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        named_params = list(self.model.named_parameters())
        nd_param_names = set([n for n, _ in named_params if any(nd in n for nd in no_decay)])
        AdamW
        param_groups = [
            {
                'params': [p for n,p in named_params if n not in nd_param_names],
                'weight_decay': self.args.get('weight_decay', 0.0),
            },
            {
                'params': [p for n,p in named_params if n in nd_param_names],
                'weight_decay': 0.0
            }
        ]
        optim_name = self.args['optim_name'].lower()
        optimizer = OPTIMIZER[optim_name](
            param_groups,
            lr = self.args['lr']
        )
        return optimizer
    
    def reset_optimizer(self):
        self.optimizer = self.get_optimizer()
    
    def get_scheduler(self, scheduler_type = None):
        steps_one_epoch = math.floor(len(self.train_dataset) / self.batch_size)
        # [TODO] support other scheduler type.
        scheduler_type = scheduler_type if scheduler_type else self.args['scheduler_type']
        if not scheduler_type or scheduler_type == 'none':
            lr_scheduler = None
        else:
            lr_scheduler = get_scheduler(
                scheduler_type, self.optimizer,
                num_warmup_steps = 0,
                num_training_steps = steps_one_epoch * self.num_epoch
            )
        return lr_scheduler
    
    def reset_scheduler(self, scheduler_type = None):
        self.lr_scheduler = self.get_scheduler(scheduler_type)
    
    def get_train_dataloader(self):
        train_dl = DataLoader(self.train_dataset,
                              batch_size = self.batch_size, 
                              shuffle = True, drop_last = True,
                              collate_fn = self.collate_fn)
        return train_dl
    
    def get_dev_dataloader(self, dataset):
        dev_dl = DataLoader(dataset,
                            batch_size = self.eval_batch_size, 
                            shuffle = False, drop_last = False,
                            collate_fn = self.collate_fn)
        return dev_dl
    
    def get_minibatch(self, batch):
        m_bs = self.batch_size // self.acc_step # mini batch size
        mbs = [] # mini batch data
        for i in range(self.acc_step):
            mb = {}
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    mb[k] = v[i*m_bs: (i+1)*m_bs]
                else:
                    mb[k] = v
            mbs.append(mb)
        return mbs
    
    def write_tb(self, data, subname, step):
        if self.writer is not None and data is not None:
            for k,v in data.items():
                self.writer.add_scalar(f'{k}/{subname}', v, step)
        
    def train(self, num_epoch = None, global_step = None, do_log_args = True):
        # log args
        if do_log_args:
            self.log_args()
        logger = self.logger
        writer = self.writer

        if global_step is not None:
            logger.info(f'Set global_step={global_step}')
            self.global_step = global_step

        # some variables
        train_dl = self.get_train_dataloader()
        iter_per_epoch = len(train_dl)
        eval_steps = self.convert_step(
            self.args['eval_steps'], iter_per_epoch)
        
        early_stop_patience = self.convert_step(
            self.args['early_stop_patience'], iter_per_epoch)

        # iter_i = 0
        model = self.model
        model.cuda()
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        step_ave = DictNumberAverager() # average at logging steps
        timestamp = time.time()
        eval_ts = time.time()
        best_iter = 0
        best_obj = 0 if self.metric_for_best else 99999
        best_metrics = None
        should_stop = False

        self.logger.info((
            f'\n\titer per epoch={iter_per_epoch}'
            f'\n\tacc step={self.acc_step}'
            f'\n\teval steps={eval_steps}'
            f'\n\tearly stop patience={early_stop_patience}'
        ))
        logger.info(f'optimizer: {optimizer.__class__}')

        num_epoch = num_epoch if num_epoch else self.num_epoch
        for epi in range(num_epoch):
            # epoch start
            # self.logger.info(f'[Epoch {epi + 1} Start]')
            for batch in train_dl:
                # batch start
                batch = to_cuda(batch)
                model.zero_grad()
                for mb in self.get_minibatch(batch):
                    # mini batch start
                    outputs = self.compute_loss(model, mb)
                    loss = outputs['loss']
                    loss = loss / self.acc_step
                    loss.backward()

                    outputs_nb = outputs_to_number(outputs)
                    step_ave.record(outputs_nb)
                # update
                # iter_i += 1
                self.global_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0) # fixed
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # print
                if self.global_step % self.logging_steps == 0:
                    logs_dt = step_ave.average()
                    # write tensorboard
                    self.write_tb(logs_dt, 'train', self.global_step)
                    # [Note] tb use glboal_step

                    # write to console and file
                    new_time = time.time()
                    duration = new_time - timestamp
                    timestamp = new_time
                    epcf = self.global_step / iter_per_epoch
                    
                    self.logger.info((
                        f'[Epoch {epcf:.2f} Iter {self.global_step}] '
                        f'{obj_to_str(logs_dt)} '
                        f'{duration:.1f}s'
                    ))
                    # debug
                    # if lr_scheduler is not None:
                    #     logger.info(str(lr_scheduler.get_last_lr()))
                
                # evaluate
                if self.global_step % eval_steps == 0:

                    epcf = self.global_step / iter_per_epoch
                    # eval dev and test set
                    dev_metrics = self.evaluate(self.dev_dataset)
                    
                    self.write_tb(dev_metrics, 'eval', self.global_step)
                    
                    test_metrics = self.evaluate_test(self.test_dataset)
                    
                    self.write_tb(test_metrics, 'test', self.global_step)
                    
                    logger.info((
                        f'Evaluation @ Epoch {epcf:.2f} Iter {self.global_step}\n'
                        f'\tDev:  {obj_to_str(dev_metrics)}\n'
                        f'\tTest: {obj_to_str(test_metrics)}'
                        ))

                    # save best model
                    obj_new = dev_metrics[self.metric_for_best]
                    is_best = self.compare_metric(obj_new, best_obj)
                    if is_best:
                        logger.info(f'New best metric {self.metric_for_best}={obj_to_str(obj_new)}({best_obj}) @ Epoch {epcf:.2f}')

                        best_obj = obj_new
                        best_iter = self.global_step
                        best_metrics = [dev_metrics, test_metrics]
                        
                        self.save_model()
                    else:
                        wait_i = self.global_step - best_iter
                        if self.early_stop and wait_i > early_stop_patience:
                            logger.info(f'No improvement after {wait_i} iter > {early_stop_patience}. Early stop.')
                            should_stop = True
                            break
                        elif self.early_stop:
                            logger.info(f'Wait {wait_i} iters < {early_stop_patience}')
                    
                    # duration from last eval finish
                    eval_dur = (time.time() - eval_ts) / 60
                    eval_ts = time.time()
                    logger.info(f'Duration: {eval_dur:.1f}min')
            if should_stop:
                break
            
        # print test metrics at best eval
        if best_metrics is not None:
            self.log_best_metrics(best_metrics,
                                  best_obj,
                                  best_iter / iter_per_epoch,
                                  best_iter)
        if writer is not None:
            writer.flush()

    def evaluate(self, dataset) -> Dict[str, Any]:
        """
        Return evaluation results in dict
        """
        if dataset is None:
            return {}
        outputs = self.do_predict(dataset)
        metrics = self.eval_metric_fn(outputs)
        return metrics
        
    def evaluate_test(self, dataset) -> Dict[str, Any]:
        return self.evaluate(dataset)

    def do_predict(self, dataset):
        """
        Get predictions on one dataest
        """
        model = self.model
        model.cuda()
        model.eval()

        batch_size = self.eval_batch_size
        use_cuda = True

        dl = DataLoader(dataset, batch_size = batch_size, 
                        shuffle = False, collate_fn = self.collate_fn)

        all_outputs = []
        with torch.no_grad():
            for batch in tqdm(dl, ncols=80, total = len(dl)):
                if use_cuda:
                    batch = to_cuda(batch)
                
                outputs = self.eval_feed_fn(model, batch)
                # outputs: List[Tensor] or Dict[str, Tensor]
                # Tensor has minimum dimensions of 1
                
                # add batch outputs
                if isinstance(outputs, (list, tuple)):
                    outputs_np = [k.cpu().numpy() for k in outputs]
                elif isinstance(outputs, dict):
                    outputs_np = {k: v.cpu().numpy() for k,v in outputs.items()}
                all_outputs.append(outputs_np)
        
        # concatenate batched results
        if isinstance(all_outputs[0], list):
            cat_outputs = []
            for i in range(len(all_outputs[0])):
                batch_out = [out[i] for out in all_outputs]
                flat_out = np.concatenate(batch_out, axis = 0)
                cat_outputs.append(flat_out)
        elif isinstance(all_outputs[0], dict):
            cat_outputs = {}
            for k in all_outputs[0].keys():
                batch_out = [out[k] for out in all_outputs]
                flat_out = np.concatenate(batch_out, axis = 0)
                cat_outputs[k] = flat_out

        return cat_outputs

    def log_args(self):
        self.logger.info('\n' + \
            json.dumps(self.args, indent = 4, sort_keys = True))
        if self.output_dir is not None:
            with open(Path(self.output_dir) / 'args.json', 'w') as f:
                json.dump(self.args, f, indent = 4, sort_keys=True)
        if self.writer is not None:
            self.writer.add_text('args', 
                json.dumps(self.args, indent = 4, sort_keys = True), 0)
    
    def log_batch_size(self, iter_per_epoch):
        self.logger.info((
            f'\n\tBatch size={self.batch_size}'
            f'\n\tIter per epoch={iter_per_epoch}'
            f'\n\tDevice batch size={self.device_bs}'
            f'\n\tAccumulation step={self.acc_step}')
        )
    
    def log_best_metrics(self, best_metrics, best_obj, best_epc, best_iter):
        best_logs = [
            f'Best {self.metric_for_best} = {obj_to_str(best_obj)} @ Epoch {best_epc:.2f} Iter {best_iter}',
            f'Dev : {obj_to_str(best_metrics[0])}',
            f'Test: {obj_to_str(best_metrics[1])}'
        ]
        self.logger.info('\n\t'+ '\n\t'.join(best_logs))
        if self.writer is not None:
            self.writer.add_text(
                'best metrics', '  \n'.join(best_logs), self.best_log_step)
            self.best_log_step += 1
    
    def compare_metric(self, obj_new, best_obj):
        if self.greater_is_better:
            return obj_new > best_obj
        else:
            return obj_new < best_obj
    
    def save_model(self):
        if self.output_dir is not None:
            save_p = Path(self.output_dir) / 'best_model.bin'
            torch.save(self.model.state_dict(), save_p)

    def convert_step(self, steps, e_iter):
        """
        positive return, negtive multiply e_iter
        """
        if steps < 0:
            steps *= -e_iter
        
        return int(steps)

    @staticmethod
    def collate_fn(features: List) -> Dict[str, Any]:
        first = features[0]
        batch = {}
        for k,v in first.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif not isinstance(v, str):
                batch[k] = torch.tensor(np.array([f[k] for f in features]))
        
        # for k,v in first.items():
        #     print(f'{k}: {v.shape}')
        # for k,v in batch.items():
        #     print(f'{k}: {v.shape}')
        # exit()

        return batch
    
    def compute_loss(self, model, batch):
        # override in subclasses
        raise NotImplementedError
    
    def eval_feed_fn(self, model, batch):
        # override in subclasses
        raise NotImplementedError
    
    def eval_metric_fn(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError