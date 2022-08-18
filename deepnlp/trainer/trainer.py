"""
Trainer for training, resuming from ckeckpoints and flow control.

Simplification of trainsformers.Trainer

Training status to save:
    trainer state
    optimizer
    lr_scheduler
    RNG
    model
    training_args
"""

import os
from re import L
import sys
import json
import logging
from textwrap import indent
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
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator
from transformers import (
    AdamW,
    get_scheduler
)
from .training_args import TrainingArguments
from .trainer_utils import (
    initialize_logger,
    set_seed,
    TrainerState,
    TrainerControl,
    NumberAverager,
    DictNumberAverager,
    torch_data_collator
)
from deepnlp.utils import concat, to_cuda
import deepnlp.utils as utils


class SimpleTrainer:
    """
    Simple trainer with default dataloader and optimizer.

    You can also customize them.
    
    Features:
        no progress bar.
    """

    PREFIX_CHECKPOINT_DIR = 'checkpoint'

    def __init__(
        self,
        model,
        args: TrainingArguments = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator if data_collator else torch_data_collator
        
        print('Build trainer...')
        # args and logger
        if args is None:
            args = TrainingArguments()
            print('Use default training arguments')
        self.args = args
        self.logger = initialize_logger(
            name = __name__,
            log_path = os.path.join(args.output_dir, 'log.txt') if args.output_dir else None
        )
        logger = self.logger

        # random seed
        set_seed(self.args.seed)

        # gpu device
        if self.args.no_cuda:
            self.device = torch.device('cpu')
            assert self.args.n_gpu == 0
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.args.n_gpu is None:
                self.args.n_gpu = torch.cuda.device_count()

        self.model_wrapped = model
        self.optimizer, self.lr_scheduler = optimizers
        # if is None, will be created during training.

        # control
        self.state = TrainerState()
        self.control = TrainerControl()
        self.loss_recoder = DictNumberAverager() # you can record more than loss

    
    def get_dataloader(self, dataset, batch_size, shuffle = False):
        dl = DataLoader(
            dataset,
            batch_size = batch_size,
            collate_fn = self.data_collator,
            shuffle = shuffle)
        
        return dl

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        If exists, do nothing. Else, a reasonable default.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps = num_training_steps, optimizer = self.optimizer)
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        Some parameters, e.g., nn.LayerNorm do not need decay. Treat them separately.
        """
        if self.optimizer is None:
            # we use AdamW here
            opt_kwargs = {
                'lr': self.args.learning_rate,
                'betas': (self.args.adam_beta1, self.args.adam_beta2),
                'eps': self.args.adam_epsilon
            }
            self.optimizer = AdamW(self.model.parameters(), **opt_kwargs)
        
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer = None):
        """
        Setup the scheduler.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def _wrap_model(self, model, training = True):
        if self.args.n_gpu > 0:
            model.cuda()
        # multi-gpu training
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        if training:
            model.train()
        else:
            model.eval()
        
        return model

    def train(self, train_dataset = None, eval_dataset = None):
        logger = self.logger

        train_dataset = train_dataset if train_dataset else self.train_dataset
        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset
        self.eval_dataset = eval_dataset # will be used in evaluate()

        train_dataloader = self.get_dataloader(
            train_dataset, self.args.train_batch_size, True)

        model = self._wrap_model(self.model)  # ! the original arg is `self.model_wrapped`. I think the current one is proper.
        if model is not self.model:
            self.model_wrapped = model

        # setup max_steps and num_train_epochs
        steps_per_epoch = len(train_dataloader)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = max_steps // steps_per_epoch + int(
                max_steps % steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.args.num_train_epochs * steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)
        
        opt_steps = max_steps // self.args.gradient_accumulation_steps
        
        self.create_optimizer_and_scheduler(num_training_steps=opt_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Training batch size = {self.args.train_batch_size}*{self.args.gradient_accumulation_steps} ({self.args.train_batch_size_per_device})")
        logger.info(f"  Total optimization steps = {opt_steps}")
        
        self.state = TrainerState(
            epoch=0,
            epoch_n = 0,
            max_steps = max_steps
        )
        self.control = TrainerControl()
        self.loss_recoder = DictNumberAverager()

        model.zero_grad()
        self.on_train_begin()

        for epoch in range(num_train_epochs):

            steps_in_epoch = len(train_dataloader)
            self.on_epoch_begin()

            for step, inputs in enumerate(train_dataloader):
                model.train()
                inputs = self._prepare_inputs(inputs)
                
                # Training step
                # Do: feed data, calculate loss, loss backward
                # Return: a dict of measurements (float) that need to be loged
                losses = self.training_step(model, inputs)
                
                self.loss_recoder.record(losses)

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch

                self.on_step_end(self.args, self.state, self.control)
                

                self._maybe_log_save_evaluate()

                if self.control.should_training_stop:
                    break
                # train step end
            
            self.state.epoch_n += 1
            self.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate()

            if self.control.should_training_stop:
                break
        
            # Train loop end
        
        self.on_train_end()
        # save best ckpt
        # create link to the checkpoint
        if self.args.output_dir is not None:
            # ver_1, save model
            """
            best_path = os.path.join(self.args.output_dir, "best_model.bin")
            logger.info(f'Save best model to {best_path}')
            torch.save(
                self.state.best_model,
                best_path)
            """
            # ver_2, save step
            pass

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs, including feedforward, compute loss, 
        loss backward, optimizer and scheduler update.

        Args:
            model (nn.Module):
                The model to train
            inputs Dict[str, Union[torch.Tensor, Any]]:
                inputs and targets of the model.
        Return:
            `Dict[str, float]`: metrics that need to be logged, e.g. loss.
        """
        

        loss = self.computing_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training.

        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        # Gradient clipping
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            model.zero_grad()

        return {'loss': loss.detach().cpu().item()}

    def computing_loss(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        return loss

    def evaluate(self, model = None, eval_dataset = None):
        """
        Predict and calculate eval metrics. Print metrics and update trainer state.
        """
        model = self.model if model is None else model
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_dataloader(eval_dataset, self.args.eval_batch_size)

        logger = self.logger
        wrap_model = self._wrap_model(model, training = False)
        time_stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        logger.info(f"{time_stamp}|***** Running evaluation {self.state.epoch_n}*****")
        logger.info(f"  Num examples = {len(eval_dataloader)}")
        logger.info(f"  Batch size = {eval_dataloader.batch_size}")

        all_logits, all_labels = self.evaluation_loop(eval_dataloader, wrap_model)

        _metrics = self.compute_metrics(all_logits, all_labels) if self.compute_metrics else {}
        metrics = {k if k.startswith('eval_') else f'eval_{k}':v for k,v in _metrics.items()}

        time_stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        self.logger.info(time_stamp + '| ' + str(metrics))

        self.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    
    def evaluation_loop(self, dataloader, model):
        """
        Get all predictions.
        """
        
        # Initialize containers
        all_logits = None
        all_labels = None
        with torch.no_grad():
            for step, inputs in enumerate(dataloader):
                inputs = self._prepare_inputs(inputs)
                logits, labels = self.prediction_step(model, inputs)
                logits = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                all_logits = logits if all_logits is None else concat(all_logits, logits)
                all_labels = labels if all_labels is None else concat(all_labels, labels)
        
        return all_logits, all_labels

            
    def prediction_step(self, model, inputs):
        """
        Predict step. Return logits and labels.
        """

        labels = inputs.pop('label')
        outputs = model(**inputs)

        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]

        return logits, labels

    def _prepare_inputs(self, inputs: dict):
        # to cuda
        if self.args.n_gpu > 0:
            return to_cuda(inputs)
        else:
            return inputs

    def _maybe_log_save_evaluate(self):
        if self.control.should_log:
            logs = self.loss_recoder.average()
            logs["learning_rate"] = self._get_learning_rate()
            self.logger.info(self.log_msg(logs))
            self.control.should_log = False
        
        if self.control.should_evaluate:
            metrics = self.evaluate()
        else:
            metrics = None
        
        if self.control.should_save:
            self._save_checkpoint(self.model, metrics = metrics)

    def _get_learning_rate(self):
        last_lr = self.lr_scheduler.get_last_lr()[0]
        return last_lr
    
    def _save_optimizer(self, output_dir):
        """
        Save optimizer and lr_scheduler
        """
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _save_checkpoint(self, model, metrics = None):
        """
        Save state_dict, optimizer, lr_scheduler, RNG, training_args
        """
        logger = self.logger

        checkpoint_folder = f"{self.PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir

        if run_dir is None:
            return None
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok= True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # save args
        utils.save_json(self.args.__dict__, os.path.join(output_dir, "training_args.json"), indent = 4)

        # save metrics
        if metrics is not None:
            with open(os.path.join(output_dir, "eval_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent = 4, ensure_ascii=False)
        
        # save optimizer and lr_scheduler
        self._save_optimizer(output_dir)

        # save trainer state
        utils.save_json(self.state.__dict__, os.path.join(output_dir, 'trainer_state.json'), indent = 4)
        
        # save RNG
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
        
        torch.save(rng_states, os.path.join(output_dir, 'rng_state.pth'))

        self.control.should_save = False

        self._rotate_checkpoints(run_dir)

    def _rotate_checkpoints(self, run_dir):
        checkpoints_sorted = self._sorted_checkpoints(run_dir)

        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return
        number_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)

        old_checkpoints = [k for k in checkpoints_sorted[:number_to_delete] if k[1]!=self.state.best_step]
        # exclude the best ckpt if it is in the old ckpts
        for checkpoint, _ in old_checkpoints:
            self.logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint)

    def _sorted_checkpoints(self, run_dir) -> List[Tuple[str, int]]:
        glob_checkpoints = [str(x) for x in Path(run_dir).glob(f"{self.PREFIX_CHECKPOINT_DIR}-*")]
        checkpoints = []
        for path in glob_checkpoints:
            regex_match = re.match(f".*{self.PREFIX_CHECKPOINT_DIR}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                checkpoints.append((path, int(regex_match.groups()[0])))
        checkpoints = sorted(checkpoints, key = lambda k: k[1]) # increasing
        return checkpoints

    def log(self, logs: Dict[str, float]) -> None:
        self.logger.info(logs)
    
    def log_msg(self, metrics):
        time_stamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
        metrics_str = [f'{k}: {v:.4g}' if v > 0.001 else f'{k}:{v:.3e}' for k,v in metrics.items()]
        metrics_str = ' '.join(metrics_str)
        msg = f'{time_stamp}| Train| {self.state.global_step} | Epoch {self.state.epoch:.3f} | {metrics_str}'
        
        return msg
    
    def on_train_begin(self):
        """
        Subclass and override for customized service
        """
        pass
        
    def on_epoch_begin(self):
        """
        Subclass and override for customized service
        """
        pass
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
            control: TrainerControl, **kwargs):
        """
        Subclass and override for customized service
        """
        # Log
        if (args.logging_steps
            and args.logging_steps > 0
            and state.global_step % (args.logging_steps * args.gradient_accumulation_steps) == 0):
            control.should_log = True
        
        # Evaluate
        # TODO: consider gradient accumulation step
        if args.eval_steps and (args.eval_steps > 0 and state.global_step % args.eval_steps == 0):
            if self.eval_dataset is not None:
                control.should_evaluate = True
            control.should_save = True
        
        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control        

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
            control: TrainerControl, **kwargs):
        """
        Subclass and override for customized service
        """
        # Evaluate
        if (
            args.eval_epochs
            and state.epoch_n % args.eval_epochs == 0
        ):
            if self.eval_dataset is not None:
                control.should_evaluate = True
        
        # Save
        if (
            args.save_epochs
            and state.epoch_n % args.save_epochs == 0
        ):
            control.should_save = True # save at each epoch.

        return control
    
    def on_train_end(self):
        """
        Subclass and override for customized service
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
            control: TrainerControl, metrics):
        """
        Perform early stop
        """
        logger = self.logger
        control.should_evaluate = False
        # Early stop
        metric_to_check = args.metric_for_best_model
        if not metric_to_check:
            if len(metrics) == 1:
                metric_to_check = list(metrics.keys())[0]
            else:
                raise ValueError(f"did not find {metric_to_check}")
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"
            )
            return
        
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or operator(metric_value, state.best_metric):
            state.best_metric = metric_value
            state.best_step = state.global_step
            #state.best_model = self.model.state_dict()
            state.early_stopping_patience_conter = 0
            
            logger.info(f'Best step {self.state.best_step}')
            if self.args.output_dir is not None:
                with open(os.path.join(self.args.output_dir, 'best_step.txt'), 'w') as f:
                    f.write(f'{self.state.best_step}\n')
                    f.write(f'{state.best_metric}')
        else:
            state.early_stopping_patience_conter += 1
        
        if args.early_stopping_patience and \
            state.early_stopping_patience_conter >= args.early_stopping_patience:
            logger.info(f'No improvement after {args.early_stopping_patience} eval steps. Early stop.')
            control.should_training_stop = True