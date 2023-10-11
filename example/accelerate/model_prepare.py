"""
Test prepare a model twice
"""
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel

model = nn.Linear(1,2)
accelerator = Accelerator()
wm_1 = accelerator.prepare(model)
wm_2 = accelerator.prepare(model)

if accelerator.is_main_process:
    print(model.weight.data)
    model.weight.data.add_(1.0)
    print(wm_2.module is model)
    print(wm_2.module.weight is model.weight)
    print(model.weight.data is wm_2.module.weight.data)
    print(wm_1.module.weight.data is wm_2.module.weight.data)
    print(wm_1.module.weight)
    print(wm_2.module.weight)