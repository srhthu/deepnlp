"""
Trainer for single GPU and multi GPUs with DistributedDataParallel

Key features:
- log, evaluate and save based on steps in the flow.
- support load from checkpoints
"""
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
from transformers.trainer_pt_utils import (
    get_model_param_count,
    nested_numpify,
    nested_concat
)

from .onegpu_utils import (
    infer_batch_sizes, to_cuda, outputs_to_number, 
    obj_to_str, DictNumberAverager, default_feed, 
    simple_data_collator, OPTIMIZER
)
from .trainer_utils import AccumulateTensors, SimpleCollator, save_dataclass_to_json
from .report import Reporter

PREFIX_CHECKPOINT_DIR = "checkpoint"
WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.json"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

@dataclass
class TrainingArgs:
    """
    General arguments for a trainer
    """
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_batch_size: Optional[int] = None

    num_train_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 1

    logging_steps: int = 10
    eval_epochs: int = 1
    eval_steps: Union[int, float] = -1
    save_ckpt: bool = True # set to False if debug
    early_stop: Optional[bool] = None
    early_stop_patience: Union[int, float] = -2

    max_grad_norm: float = 3.0

    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None

    # bellow are optional
    optim_name: Optional[str] = 'adamw'
    lr: float = 1e-5
    scheduler_type: Optional[str] = None
    save_total_limit: Optional[int] = None

    @property
    def train_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.n_gpu
    
    @property
    def eval_batch_size(self) -> int:
        return self.per_device_eval_batch_size * self.n_gpu

    @property
    def n_gpu(self):
        # do not use nn.DataParallel
        return 1
    
    @property
    def world_size(self):
        # do not use nn.DataParallel
        return 1
    
    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        return True

@dataclass
class TrainerState:
    """
    trainer state and controll
    """
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    log_history: List[Dict[str, float]] = None

    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None



class DDPTrainer:
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
        args: TrainingArgs,
        model,
        train_dataset,
        eval_dataset: Union[Dataset, Dict[str, Dataset], None] = None,
        output_dir = None,
        logger = None,
        optimizer = None,
        scheduler = None
    ):  
        self.args:TrainingArgs = args.copy() # changes on self.args do not change original args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.output_dir = output_dir

        if logger is None:
            logger = logging.getLogger(__name__)
            logger.disabled = True # Do not log. Just a plaseholder
        self.logger = logger
        self.reporter = Reporter()

        self.state = TrainerState()

        # mandatory args
        self.batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        if self.eval_batch_size is None:
            self.eval_batch_size = self.device_bs

        # self.num_epoch = self.args['num_epoch']
        # self.logging_steps = self.args['logging_steps']
        # self.eval_steps = self.args['eval_steps']
        # self.early_stop = self.args['early_stop']
        # self.early_stop_patience = self.args['early_stop_patience']

        self.metric_for_best = self.args['metric_for_best_model']
        self.greater_is_better = self.args['greater_is_better']
        if self.greater_is_better is None:
            self.greater_is_better = not 'loss' in self.metric_for_best

        self.optimizer = optimizer if optimizer else self.get_optimizer()
        self.lr_scheduler = scheduler if scheduler else self.get_scheduler()
    
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

        
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None
    ):
        args = self.args
        model = self.model
        # log args
        self.log_args()
        logger = self.logger

        # some variables
        train_dl = self.get_train_dataloader()

        # Setting up training control variables
        # max_steps: max update step
        # num_train_epochs: ceiling epochs to hold all steps
        total_train_batch_size = self.batch_size * args.gradient_accumulation_steps * args.world_size
        if hasattr(train_dl, '__len__'):
            len_dl = len(train_dl)
            num_update_steps_per_epoch = len_dl // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
        else:
            ...
        if args.eval_epochs > 0:
            args.eval_steps = args.eval_epochs * num_update_steps_per_epoch
        if args.save_epochs > 0:
            args.save_steps = args.save_epochs * num_update_steps_per_epoch

        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        model.cuda()
        model.zero_grad()
        tr_records = AccumulateTensors()
        total_batched_samples = 0
        should_stop = False
        start_time = time.time()
        for epoch in range(num_train_epochs):
            # epoch start
            # self.logger.info(f'[Epoch {epi + 1} Start]')
            for inputs in train_dl:
                total_batched_samples += 1
                # batch start
                outputs = self.training_step(model, inputs)
                tr_records.add(outputs)

                if total_batched_samples % args.gradient_accumulation_steps == 0:
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # fixed

                    # Optimizer step
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = self.state.global_step / num_update_steps_per_epoch
                
                    # Log training dynamics
                    if self.global_step % self.logging_steps == 0:
                        logs_dt = tr_records.get()
                        tr_records = AccumulateTensors()
                        # write tensorboard
                        self.do_log_step(logs_dt)
                
                    # Evaluate
                    if args.eval_steps > 0 and self.global_step % args.eval_steps == 0:
                        metrics = None
                        if self.eval_dataset is not None:
                            metrics = self.handle_evaluate()
                        # Save
                        should_stop = self.handle_save(model, metrics = metrics)
                    # End update step
                    if should_stop:
                        break
            # End epoch
            run_time = time.time() - start_time
            logger.info(str({'train_run_time': round(run_time, 2)}))
            if should_stop:
                break
        # End epoch loop
        
    def handle_evaluate(self):
        eval_dataset = self.eval_dataset
        if isinstance(self.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = self.evaluate(eval_dataset)
        return metrics

    def evaluate(self, eval_dataset, metric_key_prefix = 'eval') -> Dict[str, Any]:
        """
        Return evaluation results in dict
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        outputs = self.evaluation_loop(eval_dataloader)
        metrics = self.eval_metric_fn(outputs)
        return metrics

    def evaluation_loop(self, eval_dataloader):
        model = self.model
        model.cuda()
        model.eval()

        all_outputs: Union[Dict[str, np.ndarray], List[np.ndarray]] = None

        for inputs in eval_dataloader:
            outputs = self.prediction_step(model, inputs)
            outputs_np = nested_numpify(outputs)
            all_outputs = outputs_np if all_outputs is None else nested_concat(all_outputs, outputs_np)
        
        return all_outputs

    def prediction_step(self, model, inputs):
        inputs = to_cuda(inputs)
        with torch.no_grad():
            outputs = self.feed_eval(model, inputs)
        return outputs
    
    def feed_train(self, model, inputs):
        """Customize for subclass"""
        labels = inputs.pop('label')
        outs = model(**inputs, labels = labels)
        return {'loss': outs[0]}

    def feed_eval(self, model, inputs):
        """Customize for subclass"""
        labels = inputs.pop('label')
        outs = model(**inputs)
        logits = outs.logits
        return {'logits': logits, 'labels': labels}
    
    def handle_save(self, model, metrics = None):
        """Save model, Record best metric and handle early stop"""
        # Save
        # Save model checkpoint
        output_dir = Path(self.output_dir / f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')
        self.save_model(output_dir)

        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), output_dir / OPTIMIZER_NAME)
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), output_dir / SCHEDULER_NAME)
        
        # Update trainer state and save
        metric_to_check = self.args.metric_for_best_model
        if metrics is not None and metric_to_check is not None:
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = str(output_dir)
        if self.args.should_save:
            save_dataclass_to_json(self.state, output_dir / TRAINER_STATE_NAME)
            with open(output_dir / "metrics.json", 'w') as f:
                f.write(json.dumps(metrics, indent = 2, ensure_ascii=False) + '\n')

        if self.args.should_save:
            self._rotate_checkpoints(self.output_dir) # the run dir

    def save_model(self, output_dir: str):
        state_dict = self.model.state_dict()

        torch.save(state_dict, Path(output_dir) / WEIGHTS_NAME)

        save_dataclass_to_json(self.args, output_dir / TRAINING_ARGS_NAME)
    
    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if x.is_dir()]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        # [Note]: swap the best checkpoint to the last two position.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, output_dir):
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        checkpoints_sorted = self._sorted_checkpoints(output_dir=output_dir)
        # the best_ckpt was pushed to the 2nd last position.
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return
        
        n_ckpt_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        prev_ckpts = checkpoints_sorted[:n_ckpt_to_delete]
        # Make sure the best ckpt and latest ckpt are kept
        if prev_ckpts[-1] == self.state.best_model_checkpoint:
            prev_ckpts = prev_ckpts[:-1]
        for checkpoint in prev_ckpts:
            self.logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)
        

    def log_args(self):
        self.logger.info('\n' + \
            json.dumps(self.args, indent = 4, sort_keys = True))
    
    def compare_metric(self, obj_new, best_obj):
        if self.greater_is_better:
            return obj_new > best_obj
        else:
            return obj_new < best_obj
    
    def save_model(self):
        if self.output_dir is not None:
            save_p = Path(self.output_dir) / 'best_model.bin'
            torch.save(self.model.state_dict(), save_p)
    
    def compute_loss(self, model, batch):
        # override in subclasses
        raise NotImplementedError
    
    def eval_feed_fn(self, model, batch):
        # override in subclasses
        raise NotImplementedError
    
    def eval_metric_fn(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError