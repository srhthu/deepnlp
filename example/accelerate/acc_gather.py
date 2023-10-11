"""
Test the behavior of Accelerator.gather_for_metrics.

What if the returned number of elements does not equal to the number of input elements
"""
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

dataset = [{'x': np.array([float(i)])} for i in range(20)]
dl = DataLoader(dataset, batch_size=4)
local_rank = os.environ.get('LOCAL_RANK')
accelerator = Accelerator()

@accelerator.on_main_process
def log(obj):
    print(str(obj))

log(f'original dataloader len: {len(dl)}')

dist_dl = accelerator.prepare(dl)
log(f'after prepare: {len(dist_dl)}')

log(f'get num sample: {len(dist_dl.dataset)}')

# test two type of gather
logits_g_host = []
loss_g_host = []
logits_gfm_host = []
loss_gfm_host = []
for batch in dist_dl:
    logits = batch['x']
    loss = batch['x'].mean()
    logits_g_host.append(accelerator.gather(logits).squeeze().cpu().numpy())
    loss_g_host.append(accelerator.gather(loss).cpu().numpy())
    logits_gfm_host.append(accelerator.gather_for_metrics(logits).squeeze().cpu().numpy())
    loss_gfm_host.append(accelerator.gather_for_metrics(loss).cpu().numpy())

if accelerator.is_main_process:
    print('Rank ', local_rank)
    print(logits_g_host)
    print(loss_g_host)
    print(logits_gfm_host)
    print(loss_gfm_host)

"""
Conclusion:
gather_for_metric only works when the tensor to be gathered has same length of the inputs (i.e., batch_size)
"""