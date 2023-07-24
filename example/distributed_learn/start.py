import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

mp_vars = ['LOCAL_RANK', 'RANK', 'LOCAL_WORLD_SIZE', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
print({k:v for k,v in os.environ.items() if k in mp_vars})
print(sys.argv)
dist.init_process_group()
lr = dist.get_rank()
if lr == 0:
    print(lr)
    print(torch.cuda.device_count())

model = nn.Linear(2,3)
model.to(int(os.environ['LOCAL_RANK']))
print(model.weight.device)
"""
torchrun --nproc_per_node=2  start.py
"""