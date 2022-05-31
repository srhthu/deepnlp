# Introduction
This repository contains general modules/tools for deep learning, especially in Natural Language Processing domain.

# Main Features
The package contains these first level modules:
- trainer
  > The trainer class for training a neural network models or fine tune. It is a simplification of transformers.trainer.  
    Matched evaluation metrics
- data
  > customized torch.Dataset
- utils
  > common utils for I/O, torch.

# Trainer
## Common scenario
- Train a simple model. No save, No early stop.
- Debug, no save, evaluation, early stop.
- Train a model with save, evaluation and early stop
- Resume from a checkpoint

## Not support
- gradiant accumulation
- model parallel

## Model
- one optimizer, w or w/o scheduler
- different parameter with different optimizer
- multiple optimizer, update separately.

## Flow
- one train flow
- multiple stages, e.g., warmup. Just use train() for one stage, then total epoch will not work.

## About interval step;
eval step, log step, save(checkpoint) step  
support both epoch and step  
common case: eval and save is same.  
if log_step is float, means the ratio of log_step to num_step_per_epoch. e.g., set to 1/200, means one epoch log 200 times.  

## Note
- batch_size is for per device. the true batch_size is calculated by multiply n_gpu
- 